"""Imported by the first code cell of the ported notebook (replaces `drive.mount`).

Does only environment setup — no method code:
  * reads config.yaml (paths, seeds, device)
  * seeds python / numpy / torch and checks PYTHONHASHSEED + MPS fallback were set at kernel launch
  * forces DataLoader(num_workers=0) (notebook-defined Dataset classes cannot be pickled by
    macOS spawn workers); evaluation results do not depend on worker count
  * builds a symlink "view" reproducing Drive's flat checkpoints/ layout
  * pre-flight: refuses to run if any checkpoint is missing, because every training cell in the
    notebook is guarded only by "checkpoint exists" and would otherwise start training silently
  * writes run metadata (versions, seeds, device) next to the results
  * provides orig_subset(): the original session's id order for the `list(test_id_set)[:N]`
    subsets (Phase 4 XAI 15, Phase 4 noise-high demo 25, hallucination 60, Phase 5 80)
"""
import csv
import json
import os
import platform
import random
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repro_config import load_config  # noqa: E402

CFG = load_config()
REPRO_PATHS = CFG["paths"]
if os.environ.get("REPRO_RESULTS_ROOT"):
    REPRO_PATHS["results_root"] = Path(os.environ["REPRO_RESULTS_ROOT"]).resolve()
REPRO_SEED = CFG["seed"]
REPRO_DEVICE = torch.device(CFG["device"])

# ---- launch-time environment (must be set before the kernel starts; run.py does this) ----
if os.environ.get("PYTHONHASHSEED") != str(CFG["pythonhashseed"]):
    warnings.warn(f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')!r}, expected {CFG['pythonhashseed']}: "
                  "any remaining set-iteration order will differ from the reference runs. Launch through run.py.")
if CFG["mps_fallback"] and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
    warnings.warn("PYTORCH_ENABLE_MPS_FALLBACK is not 1: ops unsupported on MPS will raise instead of using CPU.")
if REPRO_DEVICE.type == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError("config.yaml asks for device=mps but MPS is not available")

# ---- seeds ----
random.seed(REPRO_SEED)
np.random.seed(REPRO_SEED)
torch.manual_seed(REPRO_SEED)

# ---- DataLoader worker override ----
_DataLoader = torch.utils.data.DataLoader


class _ReproDataLoader(_DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs["num_workers"] = CFG["dataloader_num_workers"]
        if REPRO_DEVICE.type != "cuda":
            kwargs["pin_memory"] = False
        super().__init__(*args, **kwargs)


torch.utils.data.DataLoader = _ReproDataLoader

# ---- checkpoint view: Drive kept classifiers, v3/ and restorers in one flat folder ----
_view = REPRO_PATHS["ckpt_view"]
_view.mkdir(parents=True, exist_ok=True)
_links = {p.name: p for p in REPRO_PATHS["classifier_ckpts"].glob("*.pt")}
_links["v3"] = REPRO_PATHS["classifier_ckpts"] / "v3"
_links.update({p.name: p for p in REPRO_PATHS["restorer_ckpts"].iterdir() if p.suffix in (".pt", ".pth")})
for _name, _target in _links.items():
    _link = _view / _name
    if not _link.exists() and not _link.is_symlink():
        _link.symlink_to(_target)

REQUIRED_CKPTS = [
    "resnet50_best.pt", "efficientnet_b3_best.pt", "vit_base_best.pt", "quality_classifier.pt",
    "v3/resnet50_v3_best.pt", "v3/efficientnet_b3_v3_best.pt", "v3/vit_base_v3_best.pt",
    "cold_diffusion_v6.pt", "swinir_gan_v6.pt", "ddpm_fundus_v1.pt", "ddpm_pathology_v1.pt",
    "RealESRGAN_x2plus.pth",
]
_missing = [c for c in REQUIRED_CKPTS if not (_view / c).exists()]
if _missing:
    raise FileNotFoundError(f"Missing checkpoints {_missing} in {_view}. Refusing to run: the notebook "
                            "would silently start training. Training needs explicit approval.")

# ---- original subset order (approved) ----
# The notebook takes subsets as list(test_id_set)[:N], whose order depends on the session's string
# hashes (Colab: random per session). The original 21 Jun session used one order for all four
# subsets: Drive's Phase 5 ensemble_predictions.csv holds its first 80 ids in processing order, and
# Drive's recovery_xai.csv (15) and ddpm_pathology_hallucination.csv (60) are exactly its prefixes.
_order_csv = REPRO_PATHS["original_results"] / "phase5_quality_ensemble" / "metrics" / "ensemble_predictions.csv"
with open(_order_csv, newline="") as _f:
    ORIG_SUBSET_ORDER = list(dict.fromkeys(r["id_code"] for r in csv.DictReader(_f)))
assert len(ORIG_SUBSET_ORDER) == 80, (len(ORIG_SUBSET_ORDER), _order_csv)


def orig_subset(id_set, n):
    """Replaces list(id_set)[:n] with the first n ids of the original session's order."""
    if n > len(ORIG_SUBSET_ORDER):
        raise ValueError(f"original order has only {len(ORIG_SUBSET_ORDER)} ids, asked for {n}")
    ids = ORIG_SUBSET_ORDER[:n]
    missing = set(ids) - set(map(str, id_set))
    if missing:
        raise ValueError(f"{len(missing)} original subset ids are not in the given id set: {sorted(missing)[:3]}")
    return ids


for _k in ("degraded_dir", "enhanced_dir", "aptos_dir"):
    if not REPRO_PATHS[_k].exists():
        raise FileNotFoundError(f"{_k} not found at {REPRO_PATHS[_k]} — see REPRODUCTION.md 'Data preparation'.")

# ---- run metadata ----
REPRO_PATHS["results_root"].mkdir(parents=True, exist_ok=True)
_pkgs = {}
for _m in ("torch", "torchvision", "timm", "numpy", "pandas", "scipy", "sklearn", "skimage", "cv2",
           "captum", "shap", "PIL", "lpips", "torchmetrics"):
    try:
        _pkgs[_m] = __import__(_m).__version__
    except Exception as _e:  # noqa: BLE001
        _pkgs[_m] = f"unavailable: {_e}"
_meta = {
    "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
    "device": str(REPRO_DEVICE), "seed": REPRO_SEED, "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
    "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
    "dataloader_num_workers": CFG["dataloader_num_workers"], "packages": _pkgs,
    "paths": {k: str(v) for k, v in REPRO_PATHS.items()},
}
(REPRO_PATHS["results_root"] / "_run_meta.json").write_text(json.dumps(_meta, indent=2))
print(f"[repro] device={REPRO_DEVICE} seed={REPRO_SEED} PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')} "
      f"results -> {REPRO_PATHS['results_root']}")
