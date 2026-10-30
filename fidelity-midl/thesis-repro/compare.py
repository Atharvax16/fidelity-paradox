"""Compare a reproduction run with the original Drive results.

    uv run python compare.py [--run-name main]

Writes results/repro/<run>/_comparison/headline.csv (+ .md) and one diff CSV per result file that
exists in both trees (all numeric columns, aligned on their key columns).
Tolerances (agreed): accuracy +-1 percentage point, correlations +-0.05.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from repro_config import load_config  # noqa: E402

ACC_TOL, CORR_TOL = 0.01, 0.05


def read(root, rel):
    p = root / rel
    return pd.read_csv(p) if p.exists() else None


def pick(df, value, **where):
    if df is None:
        return np.nan
    m = np.ones(len(df), bool)
    for k, v in where.items():
        m &= df[k].astype(str).values == str(v)
    return float(df.loc[m, value].iloc[0]) if m.any() else np.nan


def corr(root, block, metric):
    p = root / "phase4b_restoration_proof/metrics/stats/fidelity_accuracy_correlation.json"
    return json.loads(p.read_text())[block][metric][0] if p.exists() else np.nan


S2 = "phase2_model_benchmarking/metrics/stress_test_results.csv"
S3 = "phase2_model_benchmarking/metrics/v3/stress_test_results_v3.csv"
ENS = "phase2_model_benchmarking/metrics/v3/ensemble_v4_results.csv"
Q = "phase5_quality_ensemble/metrics/ensemble_summary.csv"

# (label, kind, getter(root) -> value)
HEADLINE = [
    *[(f"Clean acc, {m} (v2, TTA)", "acc", lambda r, m=m: pick(read(r, S2), "accuracy", model=m, degradation="clean"))
      for m in ("resnet50", "efficientnet_b3", "vit_base")],
    ("ViT-Base acc, noise/high (v2, TTA)", "acc", lambda r: pick(read(r, S2), "accuracy", model="vit_base", degradation="noise", level="high")),
    ("EfficientNet-B3 acc, noise/high (v2, TTA)", "acc", lambda r: pick(read(r, S2), "accuracy", model="efficientnet_b3", degradation="noise", level="high")),
    *[(f"Clean acc, {m} (v3 '{m}', TTA)", "acc", lambda r, m=m: pick(read(r, S3), "accuracy", model=m, degradation="clean"))
      for m in ("resnet50", "efficientnet_b3", "vit_base")],
    ("V4 ensemble acc, clean (7-view TTA)", "acc", lambda r: pick(read(r, ENS), "accuracy", degradation="clean")),
    ("V4 ensemble acc, noise/high", "acc", lambda r: pick(read(r, ENS), "accuracy", degradation="noise", level="high")),
    *[(f"Quality-aware routed acc, {c} (n=80)", "acc", lambda r, c=c: pick(read(r, Q), "accuracy", condition=c))
      for c in ("clean", "blur", "exposure")],
    ("Spearman rho(SSIM gain, acc delta), excl. vanilla DDPM", "corr", lambda r: corr(r, "excl_vanilla_ddpm", "ssim_gain")),
    ("Spearman rho(PSNR gain, acc delta), excl. vanilla DDPM", "corr", lambda r: corr(r, "excl_vanilla_ddpm", "psnr_gain")),
    ("Spearman rho(SSIM gain, acc delta), all restorers", "corr", lambda r: corr(r, "all", "ssim_gain")),
    ("Spearman rho(PSNR gain, acc delta), all restorers", "corr", lambda r: corr(r, "all", "psnr_gain")),
]

KEYS = ["model", "method", "degradation", "level", "variant", "condition", "kind", "id_code", "version"]


def diff_file(orig, repro, rel):
    a, b = read(orig, rel), read(repro, rel)
    keys = [k for k in KEYS if k in a.columns and k in b.columns]
    num = [c for c in a.columns if c in b.columns and c not in keys
           and pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])
           and not pd.api.types.is_bool_dtype(a[c]) and not pd.api.types.is_bool_dtype(b[c])]
    if not keys or not num or a.duplicated(keys).any() or b.duplicated(keys).any():
        return None
    for k in keys:
        a[k] = a[k].astype(str); b[k] = b[k].astype(str)
    j = a[keys + num].merge(b[keys + num], on=keys, how="outer", suffixes=("_orig", "_repro"), indicator=True)
    for c in num:
        j[f"{c}_diff"] = j[f"{c}_repro"] - j[f"{c}_orig"]
    return j, num


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-name", default="main"); a = ap.parse_args()
    cfg = load_config()
    orig = cfg["paths"]["original_results"]
    repro = cfg["paths"]["results_root"] / a.run_name
    out = repro / "_comparison"; out.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, kind, get in HEADLINE:
        o, r = get(orig), get(repro)
        d = r - o
        tol = ACC_TOL if kind == "acc" else CORR_TOL
        within = "n/a (not run)" if np.isnan(r) else ("yes" if abs(d) <= tol + 1e-9 else "NO")
        rows.append({"result": label, "original": round(o, 4), "reproduced": round(r, 4) if not np.isnan(r) else None,
                     "difference": round(d, 4) if not np.isnan(d) else None,
                     "tolerance": f"±{tol:g}", "within_tolerance": within})
    h = pd.DataFrame(rows)
    h.to_csv(out / "headline.csv", index=False)
    (out / "headline.md").write_text(h.to_markdown(index=False))
    print(h.to_string(index=False))

    summary = []
    for p in sorted(orig.rglob("*.csv")):
        rel = p.relative_to(orig)
        if not (repro / rel).exists():
            continue
        res = diff_file(orig, repro, rel)
        if res is None:
            summary.append({"file": str(rel), "note": "not comparable (no unique key)"}); continue
        j, num = res
        dst = out / "files" / rel; dst.parent.mkdir(parents=True, exist_ok=True); j.to_csv(dst, index=False)
        both = j[j["_merge"] == "both"]
        summary.append({"file": str(rel), "rows_both": len(both),
                        "rows_only_orig": int((j["_merge"] == "left_only").sum()),
                        "rows_only_repro": int((j["_merge"] == "right_only").sum()),
                        **{f"max|{c}_diff|": round(float(both[f"{c}_diff"].abs().max()), 4) for c in num[:4]}})
    pd.DataFrame(summary).to_csv(out / "file_diffs_summary.csv", index=False)
    print(f"\nper-file diffs -> {out/'file_diffs_summary.csv'}")


if __name__ == "__main__":
    main()
