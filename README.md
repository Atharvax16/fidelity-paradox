# fidelity-paradox

Does denoising retinal OCT scans erase the small lesions clinicians need to see?

This project studies the AMD-SD dataset (Hu et al., *Scientific Data* 2024) and tests hypothesis **H2**:
strong denoising (BM3D) removes small lesions, such as intraretinal fluid (IRF), more than large ones.
The test uses two measures: image contrast / contrast-to-noise ratio, and whether a U-Net segmentation
model still detects each lesion after denoising.

## Repository layout

```
AMD-SD/
  training.txt, validation.txt   official 80:20 split (eye-level, no leakage)
  _analysis/
    explore.py                   dataset statistics script (per-image / per-lesion CSVs, summary)
    explore_amd_sd.ipynb         data exploration, BM3D contrast and CNR experiments (sections 1-10)
    segment_amd_sd.ipynb         U-Net trained on original scans, evaluated on denoised scans
    *.csv, *.png, summary.txt    results and figures
```

## Data

The datasets are **not** included in this repository. Download them and place them as follows:

- **AMD-SD**: images go in `AMD-SD/images/<eye>/<eye>_<scan>.png` (1140x380: B-scan | colour mask),
  with `Demographics of the participants.xlsx` in `AMD-SD/`.
- **DDR** (diabetic retinopathy grading, optional): `DDR/`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python3 AMD-SD/_analysis/explore.py AMD-SD   # dataset statistics
cd AMD-SD/_analysis && jupyter lab            # then open the notebooks
```

- `explore_amd_sd.ipynb` section 10 runs BM3D on all 703 validation scans (about 3 hours). It is
  checkpointed through `full_val_results.csv` / `full_val_done.txt`, with denoised images cached in
  `denoised_cache/`.
- `segment_amd_sd.ipynb` writes `seg_cache/`, `seg_checkpoints/` and `seg_eval/`. These are
  git-ignored.

## Key result so far

On the full validation set, large lesions keep significantly more contrast than small ones after
BM3D, and the gap grows with sigma (`AMD-SD/_analysis/full_val_size_gap.csv`, eye-level bootstrap CIs).
