"""Execute the ported notebook from the start up to the end of a phase, in a fresh kernel.

    uv run python run.py --until phase2_benchmark [--run-name main]

* The kernel is launched with PYTHONHASHSEED and PYTORCH_ENABLE_MPS_FALLBACK from config.yaml
  (both must be set before Python/torch start, so they cannot be set from inside the notebook).
* Results go to results/repro/<run-name>/ (the notebook's RESULTS_ROOT). The notebook's own resume
  guards skip work whose output CSV already exists there, so re-running a later phase re-uses
  earlier outputs. Use a new --run-name for a from-scratch run.
* The executed notebook (with outputs), per-cell timings and any MPS->CPU fallback warnings are
  saved to runs/<run-name>/, also when a cell fails.
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from repro_config import BASE_DIR, load_config  # noqa: E402

NB = ROOT / "notebooks" / "thesis_repro.ipynb"
KERNEL = "thesis-repro"


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", required=True, help="phase tag, e.g. phase2_benchmark (see PORT_CHANGES.md)")
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--cell-timeout", type=int, default=-1, help="seconds per cell; -1 = no limit")
    a = ap.parse_args()

    nb = nbformat.read(NB, as_version=4)
    tags = [c.metadata.get("tags", [""])[0] for c in nb.cells]
    if a.until not in tags:
        sys.exit(f"unknown phase {a.until!r}; choose from {sorted(set(tags), key=tags.index)}")
    last = max(i for i, t in enumerate(tags) if t == a.until)
    nb.cells = nb.cells[: last + 1]

    results_root = (BASE_DIR / cfg["paths"]["results_root"].relative_to(BASE_DIR) / a.run_name).resolve()
    run_dir = ROOT / "runs" / a.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONHASHSEED"] = str(cfg["pythonhashseed"])
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" if cfg["mps_fallback"] else "0"
    os.environ["REPRO_RESULTS_ROOT"] = str(results_root)

    print(f"[run] cells 0-{last} (until {a.until})  results -> {results_root}")
    client = NotebookClient(nb, kernel_name=KERNEL, timeout=None if a.cell_timeout < 0 else a.cell_timeout,
                            resources={"metadata": {"path": str(ROOT / "notebooks")}}, record_timing=True)
    t0 = time.time(); err = None
    try:
        client.execute()
    except CellExecutionError as e:
        err = e
    finally:
        out_nb = run_dir / f"thesis_repro__{a.until}.ipynb"
        nbformat.write(nb, out_nb)
        timing, fallbacks = [], set()
        for i, c in enumerate(nb.cells):
            ex = c.metadata.get("execution", {})
            s, e = ex.get("iopub.execute_input"), ex.get("shell.execute_reply")
            if s and e:
                dt = (datetime.fromisoformat(e.replace("Z", "+00:00")) -
                      datetime.fromisoformat(s.replace("Z", "+00:00"))).total_seconds()
                timing.append({"cell": i, "tag": tags[i], "origin": c.metadata.get("repro_origin"),
                               "seconds": round(dt, 1)})
            for o in c.get("outputs", []):
                txt = o.get("text", "") if isinstance(o.get("text", ""), str) else "".join(o["text"])
                fallbacks.update(re.findall(r"operator '([^']+)' is not currently supported on the MPS", txt))
        (run_dir / f"timing__{a.until}.json").write_text(json.dumps(timing, indent=1))
        (run_dir / "mps_fallback_ops.txt").write_text("\n".join(sorted(fallbacks)) + ("\n" if fallbacks else ""))
        print(f"[run] {'FAILED' if err else 'done'} in {time.time()-t0:.0f}s -> {out_nb}")
        slow = sorted(timing, key=lambda r: -r["seconds"])[:5]
        print("[run] slowest cells:", ", ".join(f"{r['cell']}({r['origin']})={r['seconds']}s" for r in slow))
        if fallbacks:
            print("[run] MPS->CPU fallback ops:", ", ".join(sorted(fallbacks)))
    if err:
        print(str(err)[-3000:])
        sys.exit(1)


if __name__ == "__main__":
    main()
