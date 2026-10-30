# Reproduction of the MSc thesis pipeline (robust & explainable DR classification)

Status: **complete on CUDA (A40)** — all phases and the uplift reproduced (`--until uplift`, 30 Oct run with
the original subset order). Remaining gaps are explained in §6; open author decisions in §8.
Original notebooks are never edited: read-only copies live in `source_notebooks/` and the port is
generated from them by `tools/build_port.py` (every change is listed in `PORT_CHANGES.md`).

## 1. Environment

| | Mac (development, checks) | Cloud GPU (reproduction runs) |
|---|---|---|
| Machine | MacBook Air M5, macOS 26 (Darwin 25.4) | RunPod A40 48 GB, EU-SE-1, image `runpod/pytorch:1.4.1-cu1281-torch280-ubuntu2404` |
| Python | 3.12.13 (uv) | 3.12.3 (uv, system interpreter) |
| torch / torchvision | 2.9.1 / 0.24.1 (MPS) | 2.9.1+cu128 / 0.24.1+cu128 |
| Device settings | `device: mps`, `PYTORCH_ENABLE_MPS_FALLBACK=1`, DataLoader workers forced to 0 | `REPRO_DEVICE=cuda`, `REPRO_NUM_WORKERS=4` |

Same `uv.lock` on both machines (`uv sync --frozen`). Key pins: timm 1.0.30, numpy 2.0.2, pandas 2.2.2,
opencv 4.12.0.88, Pillow 13.0.0, scipy 1.18.1, scikit-learn 1.9.1, scikit-image 0.26.0, shap 0.54.1,
captum 0.9.0, torchmetrics 1.9.0, diffusers 0.42.0, transformers 5.20.2, basicsr 1.4.2, realesrgan 0.3.0.
The exact versions of every run are recorded in `results/repro/<run>/_run_meta.json`.

**Seeds** (`config.yaml`): `seed: 42` (Python `random`, NumPy, `torch.manual_seed`, CUDA/MPS — set in
`repro_setup.py` before any notebook code) and `PYTHONHASHSEED=0` (set by `run.py` for the kernel; it fixes
any remaining set-iteration order). The `list(set_of_ids)[:N]` subsets of Phases 4–5 no longer depend on it:
they replay the original session's order via `repro_setup.orig_subset()` (§7).

**Why the final runs are on CUDA, not MPS.** The notebook wraps inference in
`torch.cuda.amp.autocast(enabled=True)`, which is silently disabled off CUDA, so the Mac runs everything in
fp32 while the original Colab run used fp16. CUDA reproduces the original numerics; it is also ~10× faster
(full MPS run estimated at 12–14 h). Device effect measured on the checkpoint checks: at most 0.0023 in
validation accuracy between MPS and CUDA (ResNet50 v2: 0.7568 MPS vs 0.7591 CUDA = stored value).

**MPS fallback ops**: recorded per run in `runs/<run>/mps_fallback_ops.txt`.

## 2. Inputs

| Input | Location (relative to `fidelity-midl/`) | Notes |
|---|---|---|
| APTOS 2019 | `data/extracted/aptos` (from `data/Thesis/APTOS/train_images.zip`) | 3,662 images; the notebook uses `train_1.csv` (2,930) |
| EyeQ | `data/extracted/eyeq` → `data/Thesis/EYEQ` | only the CSVs are used (images only needed to retrain the quality classifier) |
| Degraded set | `data/derived/degraded` (frozen cache, 4 May) | 9 conditions × 2,930 |
| Restored set | `data/derived/enhanced` (frozen cache, 2 Jun) | 6 restorers × 9 conditions × 440 test images |
| Classifier checkpoints | `checkpoints/classifiers/checkpoints` | v2 `*_best.pt`, `v3/`, `quality_classifier.pt` |
| Restorer checkpoints | `checkpoints/restorers` | `cold_diffusion_v6`, `swinir_gan_v6`, `ddpm_fundus_v1`, `ddpm_pathology_v1`, `RealESRGAN_x2plus` |
| Original results | `results/results` | Drive `results/`, reference only, never written |

Using the frozen degraded/restored caches was agreed with the author: the original pipeline regenerated
noise without a seed on every run, so only the caches pin down a fixed input set.

## 3. Commands

### Mac
```bash
cd fidelity-midl/thesis-repro
uv sync --frozen
uv run python -m ipykernel install --sys-prefix --name thesis-repro --display-name "thesis-repro (uv)"
# one-time data layout: unzip APTOS into data/extracted/aptos, symlink EyeQ, extract both frozen caches
uv run python tools/build_port.py            # -> notebooks/thesis_repro.ipynb + PORT_CHANGES.md
uv run python checks/data_checks.py          # -> checks/out/data_checks.json
uv run python checks/checkpoint_checks.py    # -> checks/out/checkpoint_checks.json
uv run python run.py --until phase1_data --run-name main
uv run python compare.py --run-name main
```

### Cloud GPU (RunPod)
```bash
# Mac: bundle = thesis-repro (no .venv/work/runs) + checkpoints + APTOS zip/CSVs + EyeQ CSVs + frozen caches + Drive results
runpodctl send pod_bundle/fidelity-midl-bundle.tar        # sha256 95177a8a…2848a4, 11 GB
# pod
cd /workspace && runpodctl receive <code> && sha256sum fidelity-midl-bundle.tar
mkdir -p fidelity-midl && tar -xf fidelity-midl-bundle.tar -C fidelity-midl && rm fidelity-midl-bundle.tar
cd fidelity-midl && bash thesis-repro/tools/pod_setup.sh 2>&1 | tee setup.log
cd thesis-repro
REPRO_DEVICE=cuda REPRO_NUM_WORKERS=4 nohup uv run python run.py --until <phase> --run-name a40 > <phase>.log 2>&1 &
# after each phase: copy results back to the Mac
tar -czf /workspace/a40_results.tgz -C /workspace/fidelity-midl results/repro/a40 thesis-repro/runs/a40 thesis-repro/checks/out
runpodctl send /workspace/a40_results.tgz
```
`run.py` always executes cells 0 → end of the requested phase in a fresh kernel; finished cells skip work
whose output CSV already exists, so later phases do not recompute earlier ones. Cell output (including the
`[skip]` lines) only appears in the executed notebook `runs/<run>/thesis_repro__<phase>.ipynb` at the end
of the run, not in the log; check progress with `nvidia-smi` and output file timestamps.

Final run: `--until uplift --run-name a40`, 24,548 s (cell 85, Phase 4 XAI: 20,896 s; cell 95, Phase 5:
1,744 s; everything else reused or < 13 min). Executed notebook, timings and log: `runs/a40/`.

Pod pitfalls met in practice: the pod had no volume disk, so **stopping it wiped `/workspace`**
(results must be copied off before stopping); the 50 GB container disk fills up if the 11 GB bundle is
kept after unpacking.

## 4. Checks (before any reproduction run)

**Data checks** (`checks/data_checks.py`): 40 checks, 0 FAIL, 4 WARN on both machines.
- APTOS: 2,930 rows = images; `pristine_split` identical to Drive; split 2050/440/440 with test ids
  identical to Drive `test_ids.csv` (same order).
- EyeQ: label ids match the image files in the zips (Mac only; zips not uploaded to the pod).
- Degraded: all 9 conditions complete (2,930 each, same ids as the Drive manifests); blur and exposure
  regenerate from the configured parameters (parameter sweep minimum at exactly 1.0×); noise residual std
  matches the configured σ after JPEG q90.
- Restored: 6 restorers × 9 conditions × the 440 test ids.
- WARN ×3: the restored cache's **noise** images were built from a different noise draw than the frozen
  degraded cache (CLAHE recomputation differs by 10–52 grey levels vs 1.4–3.5 for blur/exposure).
- WARN ×1: Drive `phase1_summary.csv` reports 0 images (it counts `*.png` in folders of `.jpg`).

**Checkpoint checks** (`checks/checkpoint_checks.py`): 25 checks, 0 FAIL, 1 WARN. All state dicts load
with 0 missing / 0 unexpected keys. Validation accuracy recomputed vs stored (CUDA):

| Checkpoint | Actual architecture | val acc recomputed / stored | Note |
|---|---|---|---|
| v2 `resnet50_best.pt` | ResNet50 | 0.7591 / 0.7591 | |
| v2 `efficientnet_b3_best.pt` | EfficientNet-B3 | 0.7682 / 0.7682 | |
| v2 `vit_base_best.pt` | ViT-B/16 | 0.8205 / 0.8227 | |
| v3 `resnet50_v3_best.pt` | **ConvNeXt-Base** | 0.8591 / 0.8591 | κ 0.913 |
| v3 `resnet50_v3_distilled.pt` | ConvNeXt-Base | 0.7455 / 0.7455 | |
| v3 `efficientnet_b3_v3_best.pt` | **EfficientNetV2-S** | 0.4636 / 0.4636 | **WARN: undertrained, never predicts class 2** |
| v3 `efficientnet_b3_v3_distilled.pt` | EfficientNetV2-S | 0.8091 / 0.8114 | |
| v3 `vit_base_v3_best.pt` | **CLIP ViT-B/16 @384** | 0.8409 / 0.8409 | |
| `quality_classifier.pt` | ResNet18 | — | see known issues |
| `RealESRGAN_x2plus.pth` | RRDBNet | keys match (`params_ema`) | |
| `cold_diffusion_v6.pt` | | trained **15 epochs** (L1 0.0375 → 0.0069) | |

## 5. Comparison with the original results

Tolerance: accuracy ±1 percentage point, correlations ±0.05. "Original" = Drive `results/` produced by
the same checkpoints (agreed targets). Full per-row diffs: `results/repro/a40/_comparison/files/`.

| Result | Original | Reproduced (A40) | Difference | Within tolerance |
|---|---|---|---|---|
| Clean acc, ResNet50 (v2, TTA) | 0.7568 | 0.7568 | 0.0000 | yes |
| Clean acc, EfficientNet-B3 (v2, TTA) | 0.7659 | 0.7659 | 0.0000 | yes |
| Clean acc, ViT-Base (v2, TTA) | 0.8477 | 0.8477 | 0.0000 | yes |
| ViT-Base acc, noise/high (v2, TTA) | 0.6182 | 0.6091 | −0.0091 | yes |
| EfficientNet-B3 acc, noise/high (v2, TTA) | 0.2227 | 0.2273 | +0.0045 | yes |
| Clean acc, v3 "resnet50" (ConvNeXt) | 0.6386 | 0.8818 | +0.2432 | **NO — original stale** (§6.1) |
| Clean acc, v3 "efficientnet_b3" (EffV2-S) | 0.5159 | 0.4932 | −0.0227 | **NO — original stale** (§6.1) |
| Clean acc, v3 "vit_base" (CLIP ViT) | 0.6614 | 0.8432 | +0.1818 | **NO — original stale** (§6.1) |
| V4 ensemble acc, clean (7-view TTA) | 0.8773 | 0.8773 | 0.0000 | yes |
| V4 ensemble acc, noise/high | 0.3295 | 0.3273 | −0.0023 | yes |
| Quality-aware routed acc, clean (n=80) | 0.7625 | 0.7625 | 0.0000 | yes |
| Quality-aware routed acc, blur (n=80) | 0.5750 | 0.5750 | 0.0000 | yes |
| Quality-aware routed acc, exposure (n=80) | 0.6375 | 0.6750 | +0.0375 | **NO** — 3 of 80 images (§6.6) |
| Spearman ρ(SSIM gain, Δacc), excl. vanilla DDPM | −0.504 | −0.511 | −0.007 | yes |
| Spearman ρ(PSNR gain, Δacc), excl. vanilla DDPM | −0.118 | −0.150 | −0.032 | yes |
| Spearman ρ(SSIM gain, Δacc), all restorers | +0.123 | +0.119 | −0.004 | yes |
| Spearman ρ(PSNR gain, Δacc), all restorers | +0.288 | +0.269 | −0.019 | yes |
| SHAP stability (Phase 3 summary) | `summary_stability.csv` | not produced | | **not reproducible with this code base** (§6.4) |

Rows above are produced by `compare.py`. Further checks done by hand on the same run:

| Result | Original | Reproduced (A40) | Within tolerance |
|---|---|---|---|
| Phase 4 recovery accuracy (189 model × condition × variant rows) | `recovery_accuracy.csv` | 160 rows within ±1 pt, max 3.9 pts | mostly (§6.5) |
| Cold Diffusion, ViT noise/high (raw → restored) | 0.564 → 0.082 | 0.566 → 0.086 | yes |
| Phase 4 XAI, mean stability / insertion AUC per model × variant (15 images) | `recovery_xai.csv` | all 21 means within 0.031 | yes |
| Pathology-DDPM hallucination, grade-change rate (60 images × 3) | 3.3 % | 4.4 % (2 more of 180) | (§6.5) |
| Uplift A2: McNemar cells significant after Holm | 10 of 18 | 9 of 18 | (§6.7) |
| Uplift B1: fidelity-up / accuracy-down cells, 2 models | 7 of 18 | 7 of 18, same cells | yes |
| Uplift B2: EfficientNet-B3 v3 restored acc, min / mean / max | 0.045 / 0.054 / 0.098 | 0.045 / 0.054 / 0.100 | yes |
| Uplift C: held-out corruptions, fidelity-up / accuracy-down cells | 2 | 2 | yes |
| Uplift D: LPIPS vs clean (18 rows) | `lpips_mid.csv` | max diff 0.002 | yes |
| Uplift D: FID | skipped (no torchmetrics) | skipped (no torch-fidelity) | not computed in either (§8) |

Note on the thesis headline "0.814": it is not an ensemble figure. It is the EfficientNet-B3 v2
single-model clean TTA accuracy from `Thesis_optimized_final_version1`, computed with v2 checkpoints that
were overwritten on 26 May; it cannot be reproduced and was replaced by the targets above (agreed).

## 6. Gaps beyond tolerance and likely causes

### 6.1 v3 single-model stress test — original file is stale
Drive `metrics/v3/stress_test_results_v3.csv` is dated **8 May**; the v3 checkpoints are dated
**27–31 May** (`resnet50_v3_best.pt` 27 May, `efficientnet_b3_v3_best.pt` 29 May,
`vit_base_v3_best.pt` 31 May). The Drive numbers therefore come from earlier v3 models that no longer
exist. Our values equal the current checkpoints' own test accuracy from the checkpoint check
(0.8841 / 0.4932 / 0.8364 without TTA), and the V4 ensemble built from these same checkpoints reproduces
exactly — so this is a stale original, not a reproduction failure. Same cause for the large differences in
`v2_vs_v3_summary.csv`.

### 6.2 Phase 2 non-headline rows, noise (up to 4.5 points)
v2 stress test, EfficientNet-B3 noise/mid +4.5, ViT noise/mid −1.6. The original pipeline regenerated all
noisy images with **unseeded** noise on every run, so Drive's numbers come from a different noise draw
than the frozen cache. Not fixable without the original images; expected run-to-run spread.

### 6.3 Phase 2 non-headline rows, exposure (up to 2.7 points)
v2: ViT exposure/mid −2.7, exposure/low −1.4; ResNet50 exposure/low +2.0; EfficientNet-B3 exposure/high
−2.0. V4 ensemble: exposure/high −1.6, exposure/mid −1.1. Blur rows match exactly. Likely cause (not
verified): the original run regenerated the exposure images on Colab at evaluation time, while we evaluate
the frozen 4 May copies; these differ by ~1 grey level per pixel (data check 4b), which on very dark
images can flip borderline predictions.

### 6.4 Phase 3 XAI
- Insertion/deletion AUC on the progression example: 70 of 72 values within ±0.05. Exceptions: ResNet50
  blur/low deletion +0.194; ViT noise/high insertion −0.055.
- Stability on the progression example: several rows beyond ±0.05 (largest: EfficientNet-B3 noise/high
  −0.278 → +0.308). These metrics are computed on **a single image** (`d774692d9919`), and the noise rows
  depend on the noise draw (§6.2) — high variance by construction.
- `xai_results.csv`, `summary_stability.csv`, `summary_iou.csv`, `summary_insertion_auc.csv`,
  `summary_deletion_auc.csv`, `shap_quality_metrics.csv`, `shap_cross_model_consistency.csv` and the
  IG progression files are **not produced**: the Phase 3 benchmark cells are stubs in
  `Thesis_v3_restoration_proof` / `Thesis_v4_uplift`. They exist in `Thesis_optimized_final_version3`
  (Drive). Restoring them is an author decision (open). Drive's Phase 3 tables contain no attention-rollout
  rows for ViT.

### 6.5 Phase 4 restoration
- **Recovery accuracy**: 160 of 189 rows within ±1 pt; largest gap 3.9 pts (ViT CLAHE noise/high).
  Per-image predictions (`failure_mode_table.csv`, 9,240 images): deterministic restorers on blur and
  exposure flip 0 %; raw degraded inputs flip 5–8 % (regenerated at runtime in the original, §6.3);
  everything on noise differs (different noise draw, §6.2); DDPM variants flip ~5–10 %, probably from
  stochastic sampling (not verified).
- **XAI re-evaluation** (cell 85, the 15 original images): ConvNeXt and ViT with the deterministic
  restorers (CLAHE, GenAI, Cold Diffusion, SwinIR) on blur and exposure are **bit-identical** to the
  original. Differences come from noise inputs, raw degraded inputs, the stochastic DDPMs and the
  undertrained EfficientNetV2-S (only 1–24 % of its rows match exactly). Per-row median |Δ| is 0.012 for
  stability and 0.003 for insertion AUC; per model × variant means are all within 0.031.
- **Hallucination** (60 original images): 98–99 % of grades identical; grade-change rate 4.4 % vs 3.3 %.

### 6.6 Phase 5 quality-aware ensemble, exposure (+3.75 points)
Same 80 images as the original (§7). Per-image predictions agree on 100 % of clean, 96 % of blur and
92.5 % of exposure images. Clean inputs are unchanged and agree exactly; the degraded ones differ for the
reasons in §6.2–6.3, which moves 3 exposure images.

### 6.7 Uplift
- **A1/A2**: degraded rows of the ensemble-with-CI table differ by up to 1.6 pts (same numbers as Phase 2,
  §6.3); raw noise/high accuracy is 0.2955 vs 0.3205 (noise draw), giving one fewer significant McNemar
  cell (9 vs 10 of 18). Which cell changed cannot be read off: the saved original output is truncated.
- **B2**: the clean-vs-degraded EfficientNet table reprints the stale v3 stress-test file (§6.1), so it
  differs (original degraded acc 0.48–0.79, reproduced ~0.05); the restored-accuracy line matches.
- **C (held-out JPEG / motion / downscale)**: 17 of 18 rows within one image of 80. Exception: Cold
  Diffusion on JPEG, 0.30 → 0.35 (14 images change prediction, PSNR up to 0.29 dB different). The raw JPEG
  inputs are identical and Cold Diffusion is exact on motion and downscale, so the cause is not the input;
  possibly GPU numerical sensitivity (not verified). The conclusion (2 cells) is unchanged.

## 7. Known issues (recorded, not fixed)

Confirmed from the brief:
- **Cold Diffusion on noise collapses accuracy** — confirmed: ViT noise/high 0.566 raw → 0.086 restored
  (original 0.564 → 0.082); ConvNeXt 0.218 → 0.080 (original 0.218 → 0.084); EfficientNetV2-S is at
  0.052 on raw noisy input already. Note the v6 checkpoint
  trained 15 epochs; "4 epochs" was v5.
- **Quality classifier labels almost everything "reject"** — confirmed: 400/440 clean test images. Root
  cause is not (only) a domain gap: cell 49 rebinds `TFM_EVAL` to Ben-Graham 384 px preprocessing, and
  Phase 5 applies it to a ResNet18 trained on plain 224 px images. Reproduced probabilities match Drive
  `quality_clean_probs.csv` (max diff 0.0002). With the classifier's training transform the split is
  291 good / 106 usable / 43 reject. Threshold calibration masked the problem.

Found during the port:
- **A-ESRGAN never ran**: `basicsr` imports the removed `torchvision.transforms.functional_tensor`, so the
  notebook falls back to Real-ESRGAN x2 (reproduced identically). TinyU-Net is never used.
- **v3 checkpoint names are wrong**: `resnet50_v3` = ConvNeXt-Base, `efficientnet_b3_v3` = EfficientNetV2-S,
  `vit_base_v3` = CLIP ViT-B/16 @384.
- **Undertrained checkpoint used**: `efficientnet_b3_v3_best.pt` (val acc 0.46, never predicts class 2) is
  used because `USE_DISTILLED=False`, while `_read_val_kappa` (cell 42) prefers the distilled checkpoint's
  kappa for the ensemble weights. The pathology DDPM also uses the undertrained model as feature extractor.
- **Degraded images regenerated every run** with unseeded noise despite printing "[skip]" (cell 25);
  frozen degraded (4 May) and restored (2 Jun) caches use different noise draws.
- **Subsets depend on hash order**: `list(test_id_set)[:N]` (Phase 5 n=80, Phase 4 XAI 15,
  hallucination 60, V6 25, cold-diffusion demo 1) depends on the session's string hashes. **Fixed
  (approved, 28 Oct):** the original 21 Jun session used one order for all of them — Drive's Phase 5
  `ensemble_predictions.csv` lists its first 80 ids and the XAI/hallucination ids are exactly its
  prefixes — so `repro_setup.orig_subset()` replays that order (A40 run of 27 Oct still used
  `PYTHONHASHSEED=0` order: only 1/15 XAI and ~15/80 Phase 5 ids in common). The 30 Oct run covers
  exactly the original ids in all subset files (XAI 2,835/2,835 rows, hallucination 180/180, Phase 5
  240/240), which closed the earlier 5–6 pt Phase 5 gap.
- **Resume guards** (`resume-guard` in `PORT_CHANGES.md`): the slow evaluation cells reload their CSV
  when it exists; the Phase 4 XAI and Phase 5 guards also require the CSV to cover exactly the subset
  ids, so outputs from the old order are recomputed automatically.
- **Phase 5**: genai cache lookup uses `.png` but files are `.jpg` (never hits); routes are chosen by v2
  accuracy but v3 models are loaded; the baseline is v2 ViT 0.8477; the V6 `QUALITY_POLICY` patch is a
  no-op; the "ensemble" is single-model routing.
- V4 ensemble `use_tta8` actually uses 7 views; `EYEQ_CSV = next(rglob)` is nondeterministic (unused);
  uplift LPIPS noise uses unseeded `apply_degradation`.
- macOS: DataLoader `num_workers` forced to 0 (spawn cannot pickle notebook-defined classes).

## 8. Open decisions for the author
1. Restore the Phase 3 SHAP benchmark cells from `Thesis_optimized_final_version3` to reproduce SHAP
   stability / IoU / cross-model consistency?
2. Fix any of the recorded evaluation bugs (§7) for the MIDL paper? (Currently only recorded.)
3. FID (uplift D, cell 137) was skipped in both the original and this run. Computing it needs
   `torch-fidelity` added to the locked environment and a short GPU rerun of cell 137.
