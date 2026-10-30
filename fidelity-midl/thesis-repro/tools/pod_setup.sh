#!/usr/bin/env bash
# One-time setup on a Linux CUDA pod. Run from the unpacked bundle:
#     cd /workspace/fidelity-midl && bash thesis-repro/tools/pod_setup.sh
# Recreates the Mac layout (data/extracted, data/derived), installs the locked environment,
# then runs the data and checkpoint checks on CUDA.
set -euo pipefail
BASE="$(cd "$(dirname "$0")/../.." && pwd)"     # fidelity-midl/
cd "$BASE"
echo "[pod] base=$BASE"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; df -h "$BASE" | tail -1

# 1. uv + locked environment (same uv.lock as the Mac; Linux resolves the CUDA build of torch 2.9.1)
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
cd thesis-repro
uv sync --frozen
uv run python -c "import torch; print('[pod] torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run python -m ipykernel install --sys-prefix --name thesis-repro --display-name "thesis-repro (uv)"
cd "$BASE"

# 2. data layout identical to the Mac (only train_images is needed: the notebook uses train_1.csv only)
mkdir -p data/extracted/aptos/train_images data/derived
[ -n "$(ls -A data/extracted/aptos/train_images)" ] || unzip -q data/Thesis/APTOS/train_images.zip -d data/extracted/aptos/train_images
for c in train_1.csv valid.csv test.csv; do ln -sf "$BASE/data/Thesis/APTOS/$c" "data/extracted/aptos/$c"; done
ln -sfn "$BASE/data/Thesis/EYEQ" data/extracted/eyeq
[ -d data/derived/degraded/blur ] || tar -xzf data/derived/thesis_frozen/degraded/cache_degraded.tar.gz -C data/derived/
[ -d data/derived/enhanced/clahe ] || tar -xzf data/derived/thesis_frozen/enhanced/cache_enhanced_v3.tar.gz -C data/derived/
echo "[pod] aptos png: $(find data/extracted/aptos -name '*.png' | wc -l)  degraded: $(find data/derived/degraded -name '*.jpg' | wc -l)  enhanced: $(find data/derived/enhanced -name '*.jpg' | wc -l)"

# 3. checks on CUDA
cd thesis-repro
export REPRO_DEVICE=cuda REPRO_NUM_WORKERS=4
uv run python checks/data_checks.py | tail -3
uv run python -u checks/checkpoint_checks.py 2>&1 | grep -E '^\[(PASS|FAIL|WARN)|checks,'
echo "[pod] setup done. Next:  cd $BASE/thesis-repro && REPRO_DEVICE=cuda REPRO_NUM_WORKERS=4 uv run python run.py --until phase2_benchmark --run-name a40"
