"""Step 5: load every checkpoint, verify architecture/keys, sanity-predict, flag undertraining.

Run:  uv run python checks/checkpoint_checks.py
Writes checks/out/checkpoint_checks.json.

For each classifier the strongest check available is to recompute the validation metric stored
inside the checkpoint at save time (same split, same weights, same eval transform). A match means
architecture + preprocessing + weights are all wired correctly on this machine.
Model/transform code is copied verbatim from the notebook (cells 10, 14, 34-36, 38, 88).
"""
import json
import os
import sys
import warnings
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from torchvision import transforms as T  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repro_config import load_config  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
CFG = load_config(); P = CFG["paths"]; SEED = CFG["seed"]
DEVICE = torch.device(CFG["device"])
CK = P["classifier_ckpts"]; RK = P["restorer_ckpts"]; ORIG = P["original_results"]
OUT = Path(__file__).parent / "out"; OUT.mkdir(exist_ok=True)
NUM_CLASSES = 5

# ---- verbatim notebook code -------------------------------------------------------------
TIMM_NAMES = {'resnet50': 'resnet50', 'efficientnet_b3': 'efficientnet_b3', 'vit_base': 'vit_base_patch16_224'}
TFM_EVAL = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def _circle_crop(arr_rgb):
    h, w = arr_rgb.shape[:2]
    gray = cv2.cvtColor(arr_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 7, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return arr_rgb
    c = max(contours, key=cv2.contourArea)
    x, y, ww, hh = cv2.boundingRect(c)
    if ww < 50 or hh < 50:
        return arr_rgb
    return arr_rgb[y:y+hh, x:x+ww]

class BenGraham:
    def __init__(self, sigma_x=10):
        self.sigma_x = sigma_x
    def __call__(self, img_pil):
        arr  = np.array(img_pil.convert("RGB")).astype(np.float32)
        arr  = _circle_crop(arr).astype(np.float32)
        blur = cv2.GaussianBlur(arr, (0, 0), self.sigma_x)
        out  = cv2.addWeighted(arr, 4, blur, -4, 128)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

TFM_EVAL_V3 = T.Compose([BenGraham(sigma_x=10), T.Resize((384, 384)), T.ToTensor(),
                         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class MultiScaleClassifier(nn.Module):
    def __init__(self, backbone_name, num_classes=5, pretrained=True,
                 dropout=0.3, image_size=384, n_stages=4):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            feats_full = self.backbone(torch.zeros(1, 3, image_size, image_size))
        if was_training:
            self.backbone.train()
        self.n_stages = min(n_stages, len(feats_full))
        feat_dims = [f.shape[1] for f in feats_full[-self.n_stages:]]
        total_dim = sum(feat_dims)
        self.stage_attn = nn.Parameter(torch.ones(self.n_stages))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fusion = nn.Sequential(nn.Linear(total_dim, 512), nn.LayerNorm(512), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(512, num_classes))
        self.ordinal_head = nn.Linear(total_dim, 1)
        self.gradcam_target = nn.Identity()
        self._dr_arch = backbone_name
    def forward_features(self, x):
        feats = list(self.backbone(x))[-self.n_stages:]
        feats[-1] = self.gradcam_target(feats[-1])
        attn = torch.softmax(self.stage_attn, dim=0)
        return torch.cat([self.pool(f).flatten(1) * attn[i] for i, f in enumerate(feats)], dim=1)
    def forward(self, x, return_ordinal=False):
        z = self.forward_features(x)
        logits = self.fusion(z)
        return (logits, self.ordinal_head(z).squeeze(-1)) if return_ordinal else logits

BACKBONE_V3 = {"resnet50": "convnext_base.fb_in22k_ft_in1k_384",
               "efficientnet_b3": "tf_efficientnetv2_s.in21k_ft_in1k",
               "vit_base": "vit_base_patch16_clip_384.laion2b_ft_in12k_in1k"}

def stratified_split_indices(labels, val_frac=0.15, test_frac=0.15, seed=SEED):
    idx = np.arange(len(labels)); labels = np.array(labels)
    tr_idx, te_idx = train_test_split(idx, test_size=test_frac, stratify=labels, random_state=seed)
    rel_val = val_frac / (1 - test_frac)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=rel_val, stratify=labels[tr_idx], random_state=seed)
    return tr_idx.tolist(), va_idx.tolist(), te_idx.tolist()
# ------------------------------------------------------------------------------------------

aptos = P["aptos_dir"]
index = {p.stem: p for p in aptos.rglob("*.png")}
df = pd.read_csv(aptos / "train_1.csv")
_, va, te = stratified_split_indices(df["diagnosis"].astype(int).tolist())
VAL = df.iloc[va].reset_index(drop=True); TEST = df.iloc[te].reset_index(drop=True)

_cache = {}
def batch_tensors(ids, tfm, key):
    if key not in _cache:
        _cache[key] = torch.stack([tfm(Image.open(index[i]).convert("RGB")) for i in ids])
    return _cache[key]

@torch.no_grad()
def predict(model, X, bs=8):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        out.append(torch.softmax(model(X[i:i+bs].to(DEVICE)), 1).cpu())
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return torch.cat(out).numpy()

def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)

def strict_load(model, sd):
    res = model.load_state_dict(sd, strict=False)
    return list(res.missing_keys), list(res.unexpected_keys)

results = []
def record(name, status, **d):
    results.append({"check": name, "status": status, **d})
    print(f"[{status:4}] {name}  " + "  ".join(f"{k}={v}" for k, v in d.items()), flush=True)

def summarize(name, model, ckpt, X_val, y_val, X_test, y_test, ordinal=False):
    pv = predict(model, X_val); pred = pv.argmax(1)
    acc = accuracy_score(y_val, pred); f1 = f1_score(y_val, pred, average="macro")
    stored = ckpt.get("val", {})
    s_acc = float(stored.get("accuracy", float("nan")))
    pt = predict(model, X_test).argmax(1)
    dist = dict(sorted(Counter(pt.tolist()).items()))
    d = {"val_acc_recomputed": round(acc, 4), "val_acc_stored": round(s_acc, 4),
         "val_f1": round(f1, 4), "test_acc_noTTA": round(accuracy_score(y_test, pt), 4),
         "test_pred_dist": dist}
    if ordinal:
        d["val_kappa_recomputed"] = round(cohen_kappa_score(y_val, pred, weights="quadratic"), 4)
        if "kappa_qw" in stored: d["val_kappa_stored"] = round(float(stored["kappa_qw"]), 4)
    match = abs(acc - s_acc) <= 0.01
    collapsed = max(dist.values()) / len(pt) > 0.9
    undertrained = s_acc < 0.6
    status = "PASS" if match and not collapsed and not undertrained else ("FAIL" if not match else "WARN")
    flags = [f for f, c in (("collapsed_to_one_class", collapsed), ("undertrained(val_acc<0.6)", undertrained)) if c]
    record(name, status, flags=flags, **d)

y_val = VAL["diagnosis"].values; y_test = TEST["diagnosis"].values

# ---- v2 classifiers -----------------------------------------------------------------------
Xv2_val = batch_tensors(VAL.id_code, TFM_EVAL, "v2val"); Xv2_te = batch_tensors(TEST.id_code, TFM_EVAL, "v2te")
for name in TIMM_NAMES:
    ck = load(CK / f"{name}_best.pt")
    m = timm.create_model(TIMM_NAMES[name], pretrained=False, num_classes=NUM_CLASSES)
    miss, unexp = strict_load(m, ck["state_dict"])
    record(f"v2/{name}_best.pt keys", "PASS" if not miss and not unexp else "FAIL",
           arch=ck.get("arch"), missing=len(miss), unexpected=len(unexp))
    summarize(f"v2/{name}_best.pt predictions", m.to(DEVICE), ck, Xv2_val, y_val, Xv2_te, y_test)
    del m

# ---- v3 classifiers (best + distilled) -------------------------------------------------------
Xv3_val = batch_tensors(VAL.id_code, TFM_EVAL_V3, "v3val"); Xv3_te = batch_tensors(TEST.id_code, TFM_EVAL_V3, "v3te")
for name in BACKBONE_V3:
    for tag in ("best", "distilled"):
        f = CK / "v3" / f"{name}_v3_{tag}.pt"
        if not f.exists():
            record(f"v3/{f.name}", "INFO", note="not present"); continue
        ck = load(f)
        m = MultiScaleClassifier(BACKBONE_V3[name], num_classes=NUM_CLASSES, pretrained=False, image_size=384)
        miss, unexp = strict_load(m, ck["state_dict"])
        record(f"v3/{f.name} keys", "PASS" if not miss and not unexp else "FAIL",
               backbone=ck.get("backbone"), distilled_from=ck.get("distilled_from"),
               missing=len(miss), unexpected=len(unexp))
        summarize(f"v3/{f.name} predictions", m.to(DEVICE), ck, Xv3_val, y_val, Xv3_te, y_test, ordinal=True)
        del m

# ---- quality classifier (EyeQ, resnet18) ----------------------------------------------------
qsd = load(CK / "quality_classifier.pt")
qm = timm.create_model("resnet18", pretrained=False, num_classes=3)
miss, unexp = strict_load(qm, qsd)
record("quality_classifier.pt keys", "PASS" if not miss and not unexp else "FAIL", missing=len(miss), unexpected=len(unexp))
qm = qm.to(DEVICE)
orig_q = pd.read_csv(ORIG / "phase5_quality_ensemble/metrics/quality_clean_probs.csv").set_index("id_code")
ids = [i for i in TEST.id_code if i in orig_q.index]
for tag, tfm in (("TFM_EVAL (224, as trained)", TFM_EVAL), ("TFM_EVAL_V3 (BenGraham 384, as used in Phase 5)", TFM_EVAL_V3)):
    pr = predict(qm, batch_tensors(ids, tfm, f"q{tag[:12]}"))
    arg = Counter(np.array(["good", "usable", "reject"])[pr.argmax(1)].tolist())
    diff = float(np.abs(pr - orig_q.loc[ids, ["p_good", "p_usable", "p_reject"]].values).max())
    record(f"quality argmax on clean APTOS test, {tag}", "INFO",
           argmax_distribution=dict(arg), max_abs_diff_vs_drive_probs=round(diff, 4))

# ---- Real-ESRGAN: notebook loads with strict=False, so check keys explicitly ----------------
try:
    import basicsr  # noqa: F401  (fails on torchvision>=0.17, exactly as in Colab)
except Exception:
    pass
from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: E402
st = load(RK / "RealESRGAN_x2plus.pth"); sd = st.get("params_ema", st.get("params", st))
net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
miss, unexp = strict_load(net, sd)
record("RealESRGAN_x2plus.pth keys (notebook uses strict=False)", "PASS" if not miss and not unexp else "FAIL",
       key_used="params_ema" if "params_ema" in st else "params", missing=len(miss), unexpected=len(unexp))

# ---- restorer checkpoints: metadata (strict loading happens inside the notebook) ------------
for f in ("cold_diffusion_v6.pt", "swinir_gan_v6.pt", "ddpm_fundus_v1.pt", "ddpm_pathology_v1.pt"):
    ck = load(RK / f)
    meta = {k: v for k, v in ck.items() if k != "state_dict" and k != "history"}
    if "history" in ck:
        h = ck["history"]; meta.update(epochs_trained=len(h), first=h[0], last=h[-1])
    record(f"restorer {f}", "INFO", n_tensors=len(ck["state_dict"]), **meta)

json.dump(results, open(OUT / "checkpoint_checks.json", "w"), indent=2, default=str)
print(f"\n{len(results)} checks, {sum(r['status']=='FAIL' for r in results)} FAIL, "
      f"{sum(r['status']=='WARN' for r in results)} WARN -> {OUT/'checkpoint_checks.json'}")
