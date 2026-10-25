"""First-pass exploration of AMD-SD.

Each PNG is 1140x380: OCT B-scan (left 570 px) | colour mask (right 570 px).
Mask colours (Hu et al., Sci Data 2024): SRF red, IRF green, PED blue,
SHRM yellow, IS/OS (ellipsoid zone) pink, background black.

Outputs (written next to this script):
  per_image.csv      one row per B-scan: eye, split, pixel area per class, lesion counts
  per_lesion.csv     one row per connected lesion: class, area (px), bounding box
  summary.txt        dataset-level counts and checks
  lesion_sizes.png   lesion size distribution per class
Run:  python3 explore.py /path/to/AMD-SD
"""
import sys, glob, os, csv
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.dirname(os.path.abspath(__file__))
W = 570
PALETTE = {  # class -> reference RGB
    "background": (0, 0, 0),
    "SRF": (255, 0, 0),
    "IRF": (0, 255, 0),
    "PED": (0, 0, 255),
    "SHRM": (255, 255, 0),
    "ISOS": (255, 0, 255),
}
NAMES = list(PALETTE)
REF = np.array([PALETTE[k] for k in NAMES], dtype=np.int32)
LESIONS = ["SRF", "IRF", "PED", "SHRM"]  # IS/OS is a layer, analysed separately
MAX_DIST = 120  # colour distance beyond which a pixel is left unassigned

def split_of():
    s = {}
    for name, fn in [("train", "training.txt"), ("val", "validation.txt")]:
        for line in open(os.path.join(ROOT, fn)):
            line = line.strip()
            if line:
                s[os.path.basename(line)] = name
    return s

def decode(mask_rgb):
    d = ((mask_rgb[..., None, :].astype(np.int32) - REF) ** 2).sum(-1)
    lab = d.argmin(-1)
    lab[np.sqrt(d.min(-1)) > MAX_DIST] = -1
    return lab

def main():
    splits = split_of()
    files = sorted(glob.glob(os.path.join(ROOT, "images", "*", "*.png")))
    img_rows, les_rows = [], []
    unassigned_total, gray_ok = 0, 0
    for f in files:
        name = os.path.basename(f); eye = os.path.basename(os.path.dirname(f))
        a = np.asarray(Image.open(f).convert("RGB"))
        assert a.shape == (380, 2 * W, 3), (f, a.shape)
        oct_, mask = a[:, :W], a[:, W:]
        gray_ok += int(np.all(oct_[..., 0] == oct_[..., 1]) and np.all(oct_[..., 1] == oct_[..., 2]))
        lab = decode(mask)
        unassigned_total += int((lab == -1).sum())
        row = {"image": name, "eye": eye, "split": splits.get(name, "missing")}
        for ci, cname in enumerate(NAMES[1:], start=1):
            m = lab == ci
            row[f"{cname}_px"] = int(m.sum())
            if cname in LESIONS:
                comp, n = ndimage.label(m, structure=np.ones((3, 3)))
                kept = 0
                if n:
                    areas = ndimage.sum(m, comp, range(1, n + 1))
                    boxes = ndimage.find_objects(comp)
                    for k, (ar, bx) in enumerate(zip(areas, boxes), start=1):
                        if ar < 3:  # isolated speckles from compression, not lesions
                            continue
                        kept += 1
                        les_rows.append({"image": name, "eye": eye, "split": row["split"], "class": cname,
                                         "area_px": int(ar), "height_px": bx[0].stop - bx[0].start,
                                         "width_px": bx[1].stop - bx[1].start})
                row[f"{cname}_n"] = kept
        img_rows.append(row)

    with open(os.path.join(OUT, "per_image.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(img_rows[0])); w.writeheader(); w.writerows(img_rows)
    with open(os.path.join(OUT, "per_lesion.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(les_rows[0])); w.writeheader(); w.writerows(les_rows)

    # summary
    L = []
    n = len(img_rows); eyes = sorted({r["eye"] for r in img_rows}, key=int)
    L.append(f"B-scans: {n}   eyes (folders): {len(eyes)}")
    L.append(f"Left panel pure grayscale: {gray_ok}/{n}")
    L.append(f"Unassigned mask pixels (colour blends): {unassigned_total} "
             f"({100 * unassigned_total / (n * 380 * W):.4f}% of all mask pixels)")
    sp = {}
    for r in img_rows: sp.setdefault(r["split"], set()).add(r["eye"])
    L.append("Split sizes (B-scans): " + ", ".join(f"{k}={sum(r['split']==k for r in img_rows)}" for k in sp))
    shared = sp.get("train", set()) & sp.get("val", set())
    L.append(f"Eyes appearing in BOTH train and val: {len(shared)}" + (f"  e.g. {sorted(shared, key=int)[:10]}" if shared else ""))
    L.append("")
    L.append(f"{'class':6} {'scans with it':>14} {'% scans':>8} {'lesions':>8} {'median px':>10} {'<50 px':>8} {'<200 px':>8}")
    for c in LESIONS:
        has = sum(r[f"{c}_px"] > 0 for r in img_rows)
        areas = np.array([x["area_px"] for x in les_rows if x["class"] == c])
        L.append(f"{c:6} {has:14d} {100*has/n:7.1f}% {len(areas):8d} {np.median(areas) if len(areas) else 0:10.0f} "
                 f"{(areas<50).sum():8d} {(areas<200).sum():8d}")
    has = sum(r["ISOS_px"] > 0 for r in img_rows)
    L.append(f"ISOS   {has:14d} {100*has/n:7.1f}%  (layer, not counted as lesions)")
    none = sum(all(r[f'{c}_px'] == 0 for c in LESIONS) for r in img_rows)
    L.append(f"\nScans with no lesion at all: {none}")
    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))

    # figure
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=False)
    bins = np.logspace(np.log10(3), np.log10(max(x["area_px"] for x in les_rows) + 1), 30)
    for ax, c in zip(axes, LESIONS):
        areas = [x["area_px"] for x in les_rows if x["class"] == c]
        ax.hist(areas, bins=bins, color="#4a6fa5")
        ax.set_xscale("log"); ax.set_title(f"{c}  (n={len(areas)})"); ax.set_xlabel("lesion area (pixels, log)")
        ax.axvline(50, color="#c44", ls="--", lw=1)
    axes[0].set_ylabel("number of lesions")
    fig.suptitle("AMD-SD lesion size distribution (connected components per B-scan; dashed = 50 px)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "lesion_sizes.png"), dpi=130)

if __name__ == "__main__":
    main()
