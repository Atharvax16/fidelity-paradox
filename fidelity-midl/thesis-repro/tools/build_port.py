"""Build notebooks/thesis_repro.ipynb from the untouched source notebooks.

    uv run python tools/build_port.py

Composition (cell order and code unchanged except for the patches listed below):
  cells   0-118  <- source_notebooks/Thesis_v3_restoration_proof.ipynb  (full pipeline, Phases 1-5, V6-V8)
  cells 119-138  <- source_notebooks/Thesis_v4_uplift.ipynb cells 82-101 (uplift A-D)
v4 cells 0-81 are identical to the proof notebook except cell 42, where v4 deleted the call that
runs the V4 ensemble; the proof version is kept so ensemble_v4_results.csv is actually produced.

Every patch is (source cell, exact old text, new text, category, reason). The build fails if an
old text is not found exactly once, so the patch list is the complete, auditable diff.
GUARDS are expanded into the same kind of patch (category `resume-guard`) by guard_patch().
PORT_CHANGES.md is regenerated from this list.
"""
import copy
import difflib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "source_notebooks"
PROOF = SRC / "Thesis_v3_restoration_proof.ipynb"
UPLIFT = SRC / "Thesis_v4_uplift.ipynb"
OUT_NB = ROOT / "notebooks" / "thesis_repro.ipynb"
OUT_MD = ROOT / "PORT_CHANGES.md"

# phase tag for every cell of the composed notebook (index ranges are inclusive)
PHASES = [
    ("phase1_data", 0, 26),
    ("phase2_benchmark", 27, 49),
    ("phase3_xai", 50, 67),
    ("phase4_restoration", 68, 86),
    ("phase5_quality_ensemble", 87, 103),
    ("phase4b_diagnostics", 104, 118),
    ("uplift", 119, 138),
]

FULL = object()   # sentinel: replace the whole cell source

PATCHES = [
    # (cell, old, new, category, reason)
    (2, FULL,
     "# [repro] replaces `from google.colab import drive; drive.mount('/content/drive')`\n"
     "# Loads config.yaml, seeds RNGs, forces DataLoader(num_workers=0), builds the checkpoint view,\n"
     "# and refuses to run if a checkpoint is missing (missing checkpoints trigger silent training).\n"
     "import sys; sys.path.insert(0, '..')\n"
     "from repro_setup import CFG, REPRO_PATHS, REPRO_SEED, REPRO_DEVICE, orig_subset\n",
     "colab", "No Drive. Environment setup only (see repro_setup.py docstring)."),
    (3, FULL,
     "# [repro] removed `!pip install ...` — dependencies are pinned in pyproject.toml / uv.lock\n",
     "colab", "Dependencies come from the uv environment."),
    (4, "DRIVE_ROOT      = Path('/content/drive/MyDrive/Thesis')",
     "DRIVE_ROOT      = None   # [repro] no Drive; every former DRIVE_ROOT use is patched below",
     "path", "Drive root no longer exists."),
    (4, "THESIS_ZIP      = DRIVE_ROOT / 'Thesis.zip'",
     "THESIS_ZIP      = REPRO_PATHS['thesis_zip']   # [repro] reference only, never extracted",
     "path", "Data is pre-extracted once (REPRODUCTION.md)."),
    (4, "RESULTS_ROOT    = DRIVE_ROOT / 'results'",
     "RESULTS_ROOT    = REPRO_PATHS['results_root']",
     "path", "Results go to a fresh local folder; original Drive results are never written."),
    (4, "CHECKPOINTS_DIR = DRIVE_ROOT / 'checkpoints'",
     "CHECKPOINTS_DIR = REPRO_PATHS['ckpt_view']",
     "path", "Symlink view reproducing Drive's flat checkpoints/ layout."),
    (5, "LOCAL_ROOT   = Path('/content/data')",
     "LOCAL_ROOT   = REPRO_PATHS['local_root']     # [repro] was Path('/content/data')",
     "path", "Local scratch root."),
    (5, "APTOS_DIR    = LOCAL_ROOT / 'aptos'",
     "APTOS_DIR    = REPRO_PATHS['aptos_dir']      # [repro] was LOCAL_ROOT / 'aptos'",
     "path", "Pre-extracted APTOS."),
    (5, "EYEQ_DIR     = LOCAL_ROOT / 'eyeq'",
     "EYEQ_DIR     = REPRO_PATHS['eyeq_dir']       # [repro] was LOCAL_ROOT / 'eyeq'",
     "path", "EyeQ CSVs (images are only needed to retrain the quality classifier)."),
    (5, "DEGRADED_DIR = LOCAL_ROOT / 'degraded'",
     "DEGRADED_DIR = REPRO_PATHS['degraded_dir']   # [repro] was LOCAL_ROOT / 'degraded'",
     "path", "Frozen degraded set (4 May)."),
    (5, "ENHANCED_DIR = LOCAL_ROOT / 'enhanced'",
     "ENHANCED_DIR = REPRO_PATHS['enhanced_dir']   # [repro] was LOCAL_ROOT / 'enhanced'",
     "path", "Frozen restored set (2 Jun)."),
    (7, FULL,
     "# [repro] Colab extraction removed. The original smart_extract() unzipped Thesis.zip into\n"
     "# /content/data and symlinked aptos/ + eyeq/. Here the data is extracted once beforehand\n"
     "# (REPRODUCTION.md, 'Data preparation') and APTOS_DIR / EYEQ_DIR point at it.\n"
     "n_aptos = sum(1 for _ in APTOS_DIR.rglob('*.png'))\n"
     "print(f'[repro] APTOS_DIR={APTOS_DIR} ({n_aptos} png)  EYEQ_DIR={EYEQ_DIR}')\n",
     "colab", "No zip extraction at run time."),
    (9, "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
     "DEVICE = REPRO_DEVICE   # [repro] was torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
     "device", "MPS from config.yaml."),
    (23, FULL, "# [repro] removed Colab disk-usage magics (!du -sh /content/data/*, !df -h /content)\n",
     "colab", "Shell magics."),
    (24, FULL,
     "import shutil\n\n"
     "# [repro] removed `!rm -rf {DEGRADED_DIR}`: the original deleted the whole degraded set on every\n"
     "# run (then cell 25 regenerated it with fresh, unseeded noise). Never delete inputs here.\n"
     "DEGRADED_DIR.mkdir(parents=True, exist_ok=True)\n",
     "safety", "Do not delete input data."),
    (25, '_deg_cache = DRIVE_ROOT / "cache_degraded.tar.gz"',
     '_deg_cache = REPRO_PATHS["degraded_cache"]   # [repro] was DRIVE_ROOT / "cache_degraded.tar.gz"',
     "path", "Frozen degraded cache."),
    (25, '    !tar -xzf "{_deg_cache}" -C /content/data',
     '    import subprocess  # [repro] was: !tar -xzf "{_deg_cache}" -C /content/data\n'
     '    subprocess.run(["tar", "-xzf", str(_deg_cache), "-C", str(DEGRADED_DIR.parent)], check=True)',
     "colab", "Shell magic -> subprocess, same tar command."),
    (25, "for k in DEGRADATION_TYPES:\n    for l in DEGRADATION_LEVELS:\n        m = build_split_parallel(k, l)\n"
         "        print(f'  {k}/{l}: {len(m)} images')\n\n!df -h /content\"",
     "if not _degraded_present:   # [repro] honour the guard above (approved: use the frozen degraded set)\n"
     "    for k in DEGRADATION_TYPES:\n        for l in DEGRADATION_LEVELS:\n            m = build_split_parallel(k, l)\n"
     "            print(f'  {k}/{l}: {len(m)} images')\n# [repro] removed malformed magic: !df -h /content\"",
     "logic-approved",
     "Original printed '[skip] not regenerating' and then regenerated all 26,370 images anyway, with "
     "unseeded noise. Guarded so the frozen set is used (user-approved)."),
    (70, "%pip install -q basicsr realesrgan facexlib gfpgan",
     "# [repro] removed `%pip install -q basicsr realesrgan facexlib gfpgan` (pinned in uv.lock)",
     "colab", "Dependencies come from the uv environment."),
    (82, "_ENH_CACHE = Path('/content/drive/MyDrive/Thesis/cache_enhanced_v3.tar.gz')",
     "_ENH_CACHE = REPRO_PATHS['enhanced_cache']   # [repro] was Path('/content/drive/MyDrive/Thesis/cache_enhanced_v3.tar.gz')",
     "path", "Frozen enhanced cache (exists -> the cell skips re-tarring)."),
    (99, "    dest = Path(f'/content/drive/MyDrive/Thesis/{name}')",
     "    dest = REPRO_PATHS[f'{name}_dir']   # [repro] was the Drive mirror; points at the local copy -> no-op",
     "path", "Drive backup copy becomes a no-op."),
    (73, "test_id = list(test_id_set)[0]",
     "test_id = orig_subset(test_id_set, 1)[0]   # [repro] was list(test_id_set)[0]",
     "logic-approved", "Subset order (see Phase 4 XAI patch); demo image only."),
    (85, "EXPLAIN_IDS_P4 = list(test_id_set)[:15]",
     "EXPLAIN_IDS_P4 = orig_subset(test_id_set, 15)   # [repro] was list(test_id_set)[:15]",
     "logic-approved", "Subset order: list(set)[:N] depends on per-session string hashes; use the original "
     "session's ids (repro_setup.orig_subset, user-approved)."),
    (95, "for id_code in tqdm(list(test_id_set)[:80], desc='ensemble'):",
     "for id_code in tqdm(orig_subset(test_id_set, 80), desc='ensemble'):   # [repro] was list(test_id_set)[:80]",
     "logic-approved", "Subset order (see Phase 4 XAI patch)."),
    (105, "for cand in list(test_id_set)[:25]:",
     "for cand in orig_subset(test_id_set, 25):   # [repro] was list(test_id_set)[:25]",
     "logic-approved", "Subset order (see Phase 4 XAI patch)."),
    (110, "_sample = set(map(str, list(test_id_set)[:60]))",
     "_sample = set(orig_subset(test_id_set, 60))   # [repro] was set(map(str, list(test_id_set)[:60]))",
     "logic-approved", "Subset order (see Phase 4 XAI patch)."),
]

# Resume guards: (cell, first line, last line, csv expr, extra validity condition or None, skip code, reason).
# The block from `first` to `last` (matched on stripped text, each exactly once) is moved under
# `else:`; when the CSV already exists (and passes the validity check) the skip code reloads the
# variables later cells read instead. Applied after PATCHES, so blocks contain the patched code.
_IDS = "dtype={'id_code': str}"   # hex ids like '7e20…' would otherwise parse as floats
GUARDS = [
    (32, "rows = []", "pivot.to_csv(P2 / 'metrics' / 'accuracy_pivot.csv')",
     "P2 / 'metrics' / 'stress_test_results.csv'", None,
     "stress_df = pd.read_csv(_guard_csv)\n"
     "pivot = stress_df.pivot_table(index=['degradation', 'level'], columns='model', values='accuracy').round(3)",
     "v2 stress test (~2 min on A40)."),
    (42, "rows_ens = []", 'print("Saved:", ens_csv)',
     "P2 / 'metrics' / 'v3' / 'ensemble_v4_results.csv'", None,
     "ens_csv = _guard_csv\nens_df_v4 = pd.read_csv(ens_csv)",
     "V4 ensemble (~10 min)."),
    *[(c, "metric_rows = []", f"{k}_metrics.to_csv(P3 / 'metrics' / f'{k}_progression_metrics_{{demo_id}}.csv', index=False)",
       f"P3 / 'metrics' / f'{k}_progression_metrics_{{demo_id}}.csv'", None,
       f"{k}_metrics = pd.read_csv(_guard_csv)\nprint({k}_metrics.to_string(index=False))",
       f"Phase 3 {k} progression (~2.5 min; the figure is not redrawn on skip).")
      for c, k in ((62, "blur"), (63, "exposure"), (64, "noise"))],
    (83, "rows = []", "rec_df.to_csv(P4 / 'metrics' / 'recovery_accuracy.csv', index=False)",
     "P4 / 'metrics' / 'recovery_accuracy.csv'", None,
     "rec_df = pd.read_csv(_guard_csv)",
     "Phase 4 recovery accuracy (~11 min)."),
    (85, "rows = []", "xai_rec.to_csv(P4 / 'metrics' / 'recovery_xai.csv', index=False)",
     "P4 / 'metrics' / 'recovery_xai.csv'",
     f"set(pd.read_csv(_guard_csv, {_IDS})['id_code']) == set(EXPLAIN_IDS_P4)",
     f"xai_rec = pd.read_csv(_guard_csv, {_IDS})",
     "Phase 4 XAI (~4.5 h). Reused only if the CSV covers exactly EXPLAIN_IDS_P4."),
    (95, "rows = []", "ens_df.to_csv(P5 / 'metrics' / 'ensemble_predictions.csv', index=False)",
     "P5 / 'metrics' / 'ensemble_predictions.csv'",
     f"set(pd.read_csv(_guard_csv, {_IDS})['id_code']) == set(orig_subset(test_id_set, 80))",
     f"ens_df = pd.read_csv(_guard_csv, {_IDS})",
     "Phase 5 routed pipeline (~23 min). Reused only if the CSV covers exactly the 80 subset ids."),
    (108, "failure_rows = []", "failure_df.to_csv(P4 / 'metrics' / 'failure_mode_table.csv', index=False)",
     "P4 / 'metrics' / 'failure_mode_table.csv'", None,
     f"failure_df = pd.read_csv(_guard_csv, {_IDS})",
     "Failure-mode table (~7 min)."),
]


def guard_patch(text, first, last, csv_expr, valid, on_skip):
    """Return (old, new) that moves lines first..last of `text` under a resume guard."""
    lines = text.splitlines()
    i = [n for n, l in enumerate(lines) if l.strip() == first]
    j = [n for n, l in enumerate(lines) if l.strip() == last]
    if len(i) != 1 or len(j) != 1 or j[0] < i[0]:
        raise SystemExit(f"guard markers not unique/ordered: {first!r} {len(i)}, {last!r} {len(j)}")
    i, j = i[0], j[0]
    ind = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    cond = "_guard_csv.exists()" + (f" and {valid}" if valid else "")
    new = [f"{ind}_guard_csv = {csv_expr}   # [repro] resume guard: reuse the CSV instead of recomputing",
           f"{ind}if {cond}:",
           f"{ind}    print(f'[skip] {{_guard_csv}} exists -> loaded, not recomputed')",
           *[f"{ind}    {l}" for l in on_skip.splitlines()],
           f"{ind}else:",
           *[f"    {l}" if l.strip() else l for l in lines[i:j + 1]]]
    return "\n".join(lines[i:j + 1]), "\n".join(new)


def src(cell):
    return "".join(cell["source"])


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def main():
    proof = json.loads(PROOF.read_text())
    uplift = json.loads(UPLIFT.read_text())
    assert len(proof["cells"]) == 119, len(proof["cells"])
    assert len(uplift["cells"]) == 102, len(uplift["cells"])

    cells = [copy.deepcopy(c) for c in proof["cells"]] + [copy.deepcopy(c) for c in uplift["cells"][82:102]]
    origin = [("proof", i) for i in range(119)] + [("uplift", i) for i in range(82, 102)]
    original_text = [src(c) for c in cells]

    def apply(idx, old, new):
        s = src(cells[idx])
        if old is FULL:
            set_src(cells[idx], new)
        else:
            n = s.count(old)
            if n != 1:
                raise SystemExit(f"patch on cell {idx} expected 1 match, found {n}: {old[:60]!r}")
            set_src(cells[idx], s.replace(old, new))

    applied = []
    for idx, old, new, cat, why in PATCHES:
        apply(idx, old, new); applied.append((idx, old, new, cat, why))
    for idx, first, last, csv_expr, valid, on_skip, why in GUARDS:
        old, new = guard_patch(src(cells[idx]), first, last, csv_expr, valid, on_skip)
        apply(idx, old, new); applied.append((idx, old, new, "resume-guard", why))

    # no Colab residue may survive
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        # ignore comments: [repro] comments quote the removed code on purpose
        code = "\n".join(l.split("#", 1)[0] for l in src(c).splitlines())
        bad = [l for l in code.splitlines() if re.match(r"\s*[!%]", l)] + \
              [t for t in ("/content", "google.colab", "drive.mount") if t in code]
        if bad:
            raise SystemExit(f"Colab residue in cell {i}: {bad[:3]}")

    for name, a, b in PHASES:
        for i in range(a, b + 1):
            cells[i].setdefault("metadata", {})["tags"] = [name]
    for i, c in enumerate(cells):
        c.setdefault("metadata", {})["repro_origin"] = f"{origin[i][0]}[{origin[i][1]}]"
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None

    nb = copy.deepcopy(proof)
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "thesis-repro", "display_name": "thesis-repro (uv)", "language": "python"}
    nb["metadata"].pop("accelerator", None); nb["metadata"].pop("colab", None)
    OUT_NB.parent.mkdir(exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False))

    # ---- PORT_CHANGES.md ----
    lines = ["# Port changes (generated by tools/build_port.py — do not edit by hand)", "",
             "Composed notebook: `notebooks/thesis_repro.ipynb`, "
             f"{len(cells)} cells = `Thesis_v3_restoration_proof.ipynb` cells 0–118 + "
             "`Thesis_v4_uplift.ipynb` cells 82–101. All other cells are byte-identical to their source.", "",
             "| # | cell | origin | category | reason |", "|---|---|---|---|---|"]
    for n, (idx, old, new, cat, why) in enumerate(applied, 1):
        lines.append(f"| {n} | {idx} | {cells[idx]['metadata']['repro_origin']} | {cat} | {why} |")
    lines += ["", "Categories: `colab`/`path`/`device` = setup only; `safety` = prevents data deletion; "
              "`logic-approved` = control-flow change approved by the author; `resume-guard` = skip a "
              "computation whose output CSV already exists and reload it (no effect on a fresh run).", "",
              "Phase tags (used by `run.py --until`):", ""]
    lines += [f"- `{name}`: cells {a}–{b}" for name, a, b in PHASES]
    lines += ["", "## Full diff", ""]
    for i in sorted({p[0] for p in applied}):
        d = difflib.unified_diff(original_text[i].splitlines(), src(cells[i]).splitlines(),
                                 f"source {cells[i]['metadata']['repro_origin']}", f"ported cell {i}",
                                 lineterm="", n=1)
        lines += ["```diff", *d, "```", ""]
    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_NB} ({len(cells)} cells, {len(applied)} patches) and {OUT_MD}")


if __name__ == "__main__":
    main()
