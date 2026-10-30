"""Load config.yaml and resolve paths. Shared by the ported notebook, run.py and checks/.

Environment overrides (so the same checkout runs on the Mac and on a CUDA pod without edits):
  REPRO_DEVICE=cuda        overrides `device`
  REPRO_NUM_WORKERS=4      overrides `dataloader_num_workers` (Linux fork can pickle notebook classes)
"""
import os
from pathlib import Path

import yaml

REPO_DIR = Path(__file__).resolve().parent          # thesis-repro/
BASE_DIR = REPO_DIR.parent                          # fidelity-midl/


def load_config(path=REPO_DIR / "config.yaml"):
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["paths"] = {k: (BASE_DIR / v).resolve() if not Path(v).is_absolute() else Path(v)
                    for k, v in cfg["paths"].items()}
    if os.environ.get("REPRO_DEVICE"):
        cfg["device"] = os.environ["REPRO_DEVICE"]
    if os.environ.get("REPRO_NUM_WORKERS"):
        cfg["dataloader_num_workers"] = int(os.environ["REPRO_NUM_WORKERS"])
    return cfg
