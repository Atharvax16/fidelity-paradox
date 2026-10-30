"""Step 4: explicit data checks for the silent failures seen in the original pipeline.

Run:  uv run python checks/data_checks.py
Writes checks/out/data_checks.json and prints a PASS/FAIL/WARN line per check.
Degradation / CLAHE / split functions are copied verbatim from the notebook (cells 12, 22, 68)
so the checks do not depend on executing the notebook.
"""
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repro_config import load_config  # noqa: E402

CFG = load_config()
P = CFG["paths"]
SEED = CFG["seed"]
ORIG = P["original_results"]
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

KINDS, LEVELS = ("blur", "exposure", "noise"), ("low", "mid", "high")
DEGRADATION_PARAMS = {  # notebook cell 9
    "blur": {"low": 2.0, "mid": 5.0, "high": 9.0},
    "exposure": {"low": 0.7, "mid": 0.4, "high": 0.2},
    "noise": {"low": 0.02, "mid": 0.06, "high": 0.12},
}
ENHANCERS = ("clahe", "genai", "cold_diff", "swinir_gan", "ddpm", "ddpm_path")


# ---- verbatim from notebook cell 12 ----
def stratified_split_indices(labels, val_frac=0.15, test_frac=0.15, seed=SEED):
    idx = np.arange(len(labels)); labels = np.array(labels)
    tr_idx, te_idx = train_test_split(idx, test_size=test_frac, stratify=labels, random_state=seed)
    rel_val = val_frac / (1 - test_frac)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=rel_val,
                                       stratify=labels[tr_idx], random_state=seed)
    return tr_idx.tolist(), va_idx.tolist(), te_idx.tolist()


# ---- verbatim from notebook cell 22 (noise omitted: unseeded) ----
def _np(img):  return np.array(img.convert('RGB'))
def _pil(arr): return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
def gaussian_blur(img, sigma):
    k = max(3, int(2 * round(3 * sigma) + 1))
    return _pil(cv2.GaussianBlur(_np(img), (k, k), sigmaX=sigma, sigmaY=sigma))
def exposure_shift(img, gain): return _pil(_np(img).astype(np.float32) * gain)


# ---- verbatim from notebook cell 68 ----
_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
def enhance_clahe(img):
    arr = np.array(img.convert('RGB'))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    lab[..., 0] = _clahe.apply(lab[..., 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


results = []
def record(name, status, **detail):
    results.append({"check": name, "status": status, **detail})
    extra = "  ".join(f"{k}={v}" for k, v in detail.items())
    print(f"[{status:4}] {name}  {extra}")


def jpeg_roundtrip(img, **kw):
    import io
    buf = io.BytesIO(); img.save(buf, format="JPEG", **kw); buf.seek(0)
    return Image.open(buf).convert("RGB")


def mad(a, b):
    return float(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).mean())


# 1. APTOS counts / labels -------------------------------------------------------------
aptos = P["aptos_dir"]
index = {p.stem: p for p in aptos.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")}
df = pd.read_csv(aptos / "train_1.csv")
missing = [i for i in df["id_code"] if i not in index]
labels = dict(sorted(Counter(df["diagnosis"]).items()))
record("aptos.train_1 rows == images", "PASS" if len(df) == 2930 and not missing else "FAIL",
       rows=len(df), indexed_total=len(index), missing=len(missing), labels=labels)
orig_pristine = pd.read_csv(ORIG / "phase1_data_engineering/metrics/pristine_split.csv")
same = orig_pristine["id_code"].tolist() == df["id_code"].tolist() and \
       orig_pristine["diagnosis"].tolist() == df["diagnosis"].tolist()
record("aptos.pristine_split identical to Drive", "PASS" if same else "FAIL")

# 2. split reproduces the original 2050/440/440 and the saved test ids ----------------
tr, va, te = stratified_split_indices(df["diagnosis"].astype(int).tolist())
test_ids = [df.iloc[i]["id_code"] for i in te]
orig_test = pd.read_csv(ORIG / "phase2_model_benchmarking/metrics/test_ids.csv")["id_code"].tolist()
record("split sizes 2050/440/440", "PASS" if (len(tr), len(va), len(te)) == (2050, 440, 440) else "FAIL",
       sizes=(len(tr), len(va), len(te)),
       test_labels=dict(sorted(Counter(df.iloc[te]["diagnosis"]).items())))
record("split test ids == Drive test_ids.csv (same order)", "PASS" if test_ids == orig_test else "FAIL",
       n_overlap=len(set(test_ids) & set(orig_test)))
test_set = set(test_ids)

# 3. EyeQ ids vs files (zip listings; images are not extracted) -----------------------
eyeq = P["eyeq_dir"]
for split, zname in (("train", "train_images_768.zip"), ("test", "test_images_768.zip")):
    lab = pd.read_csv(eyeq / f"Label_EyeQ_{split}.csv")
    if not (eyeq / zname).exists():   # e.g. on a cloud pod where the 89 GB of EyeQ zips are not uploaded
        record(f"eyeq.{split} label ids match image files", "INFO", note=f"{zname} not present; checked on the Mac")
        continue
    names = {Path(n).stem for n in zipfile.ZipFile(eyeq / zname).namelist()
             if n.lower().endswith((".png", ".jpg", ".jpeg"))}
    ids = lab["image"].astype(str).map(lambda s: Path(s).stem)
    n_match = int(ids.isin(names).sum())
    # the original filter_pristine joined EyeQ onto APTOS id_code -> report that overlap too
    aptos_overlap = len(set(ids) & set(df["id_code"]))
    record(f"eyeq.{split} label ids match image files",
           "PASS" if n_match == len(lab) else "FAIL",
           rows=len(lab), matched=n_match, quality=dict(sorted(Counter(lab["quality"]).items())),
           overlap_with_aptos_ids=aptos_overlap)

# 4. degraded set: 9 conditions x 2930, manifests consistent ---------------------------
deg = P["degraded_dir"]
for k in KINDS:
    for l in LEVELS:
        d = deg / k / l
        man = pd.read_csv(d / "manifest.csv")
        files = {p.name for p in d.glob("*.jpg")}
        ok = len(man) == 2930 and set(man["rel_path"]) == files and set(man["id_code"]) == set(df["id_code"])
        orig_man = pd.read_csv(ORIG / f"phase1_data_engineering/metrics/manifest_{k}_{l}.csv")
        record(f"degraded.{k}/{l} complete", "PASS" if ok else "FAIL",
               manifest=len(man), files=len(files),
               same_ids_as_drive_manifest=set(orig_man["id_code"]) == set(man["id_code"]))

# 4b. frozen blur/exposure == deterministic regeneration (proves params unchanged since 4 May).
# Control: the same comparison with the parameter nudged by 25% must be clearly larger, otherwise
# the check has no power. Residual differences come from libjpeg/OpenCV version drift.
rng = np.random.default_rng(SEED)
sample = rng.choice(sorted(test_set), 40, replace=False)
DEG = {"blur": gaussian_blur, "exposure": exposure_shift}
SWEEP = (0.8, 0.9, 1.0, 1.1, 1.25)       # multiples of the configured parameter
for k in ("blur", "exposure"):
    for l in LEVELS:
        p = DEGRADATION_PARAMS[k][l]
        err = {f: [] for f in SWEEP}
        for sid in sample[:20]:
            img = Image.open(index[sid]).convert("RGB").resize((224, 224), Image.BILINEAR)
            frozen = Image.open(deg / k / l / f"{sid}.jpg").convert("RGB")
            for f in SWEEP:
                err[f].append(mad(jpeg_roundtrip(DEG[k](img, p * f), quality=90, optimize=False), frozen))
        err = {f: round(float(np.mean(v)), 3) for f, v in err.items()}
        best = min(err, key=err.get)
        record(f"degraded.{k}/{l} frozen == regenerated", "PASS" if best == 1.0 and err[1.0] < 2.0 else "FAIL",
               param=p, mad_by_param_multiple=err)

# 4c. noise: residual std vs what the notebook's own operator produces after JPEG q90 + clipping
# (simulated with a fixed seed), not vs the raw requested std.
sim_rng = np.random.default_rng(SEED)
for l in LEVELS:
    stds, sim = [], []
    for sid in sample:
        clean_img = Image.open(index[sid]).convert("RGB").resize((224, 224), Image.BILINEAR)
        clean = np.asarray(clean_img, np.float32)
        noisy = np.asarray(Image.open(deg / "noise" / l / f"{sid}.jpg").convert("RGB"), np.float32)
        s = DEGRADATION_PARAMS["noise"][l] * 255
        simulated = _pil(clean + sim_rng.normal(0, s, clean.shape))
        simulated = np.asarray(jpeg_roundtrip(simulated, quality=90, optimize=False), np.float32)
        inside = clean.mean(-1) > 20          # ignore the black border where clipping at 0 eats noise
        stds.append((noisy - clean)[inside].std()); sim.append((simulated - clean)[inside].std())
    got, exp = float(np.mean(stds)), float(np.mean(sim))
    record(f"degraded.noise/{l} residual std", "PASS" if abs(got - exp) / exp < 0.10 else "FAIL",
           requested=round(DEGRADATION_PARAMS['noise'][l] * 255, 2),
           expected_after_jpeg=round(exp, 2), measured=round(got, 2))

# 5. enhanced set: 6 restorers x 9 conditions x 440 test ids ---------------------------
enh = P["enhanced_dir"]
for e in ENHANCERS:
    bad = []
    for k in KINDS:
        for l in LEVELS:
            d = enh / e / k / l
            files = {p.stem for p in d.glob("*.jpg")}
            if files != test_set: bad.append(f"{k}/{l}:{len(files)}")
    record(f"enhanced.{e} 9x440 == test ids", "PASS" if not bad else "FAIL", bad=bad)

# 5b. was the enhanced cache built from THIS degraded cache? (CLAHE is deterministic)
for k in KINDS:
    for l in LEVELS:
        diffs = []
        for sid in sample[:20]:
            src = Image.open(deg / k / l / f"{sid}.jpg").convert("RGB")
            re_ = jpeg_roundtrip(enhance_clahe(src))           # PIL .save('x.jpg') default quality 75
            diffs.append(mad(re_, Image.open(enh / "clahe" / k / l / f"{sid}.jpg").convert("RGB")))
        m = float(np.mean(diffs))
        # blur/exposure are deterministic, so they calibrate what "same source" looks like (~1-4)
        record(f"enhanced.clahe/{k}/{l} built from frozen degraded", "PASS" if m < 5.0 else "WARN",
               mean_abs_diff_0_255=round(m, 3))

# 6. original phase1 summary (counts '*.png' in a folder of .jpg files) ----------------
s = pd.read_csv(ORIG / "phase1_data_engineering/metrics/phase1_summary.csv")
record("drive phase1_summary.csv n_images", "WARN" if (s["n_images"] == 0).all() else "PASS",
       n_images=s["n_images"].tolist())

json.dump(results, open(OUT / "data_checks.json", "w"), indent=2, default=str)
n_fail = sum(r["status"] == "FAIL" for r in results)
print(f"\n{len(results)} checks, {n_fail} FAIL, {sum(r['status']=='WARN' for r in results)} WARN "
      f"-> {OUT/'data_checks.json'}")
sys.exit(1 if n_fail else 0)
