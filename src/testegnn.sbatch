#!/usr/bin/env bash
#SBATCH --account=bdbq-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --constraint=scratch&projects
#SBATCH --job-name=test_egnn
#SBATCH --output=slurm-test-egnn-%j.out
#SBATCH --error=slurm-test-egnn-%j.err

set -euo pipefail

# --- config you can tweak ---
WORKSPACE="/scratch/bdbq/mcampos1/eigenEGNN_model"     # repo root
DATA_NAME="test"                                       # train.py expects processed_${DATA_NAME}.pt
PT_DIR="${WORKSPACE}/data_processed"
WALL_EPOCHS=2                                          # small smoke test
BATCH=2                                                # tiny batch is fine for sanity run
LR=1e-3
LOSS="L2"
SCHED="Cos"
HIDDEN="64 128 256"                                    # keep default EGNN hidden sizes
MLP_HID=64
MLP_LAYERS=2

echo "[$(date)] Node: $(hostname)"
echo "CUDA visible: ${CUDA_VISIBLE_DEVICES:-unset}"

module reset
source /sw/external/python/anaconda3_gpu/bin/activate
conda activate /projects/bdbq/eigenvenv

# Unbuffered python for live logs
export PYTHONUNBUFFERED=1
# Optional: keep PyTorch from grabbing too many threads
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Move to src so train.py is importable the way your tree expects
cd "$SLURM_SUBMIT_DIR" || { echo "SLURM_SUBMIT_DIR missing"; exit 1; }
if [[ -f train.py ]]; then
  : # already in src
elif [[ -d src && -f src/train.py ]]; then
  cd src
else
  echo "Could not locate src/train.py from $(pwd)"; exit 1
fi
echo "CWD: $(pwd)"

# Ensure the dataset filename matches train.py's expectation:
# train.py looks for: ${WORKSPACE}/data_processed/processed_${DATA_NAME}.pt
mkdir -p "${PT_DIR}"
if [[ -f "${PT_DIR}/test.pt" && ! -f "${PT_DIR}/processed_${DATA_NAME}.pt" ]]; then
  echo "[info] Creating symlink processed_${DATA_NAME}.pt -> test.pt"
  ln -sfn "${PT_DIR}/test.pt" "${PT_DIR}/processed_${DATA_NAME}.pt"
fi

echo "[info] Available .pt files:"
ls -lh "${PT_DIR}"/*.pt || true

echo
echo "=== Running training smoke test ==="
set -x
python -u train.py \
  --workspace "${WORKSPACE}" \
  --output_dir "models" \
  --datasets "${DATA_NAME}" \
  --epochs "${WALL_EPOCHS}" \
  --batch "${BATCH}" \
  --lr "${LR}" \
  --loss "${LOSS}" \
  --scheduler "${SCHED}" \
  --hidden ${HIDDEN} \
  --mlp-hidden "${MLP_HID}" \
  --mlp-layers "${MLP_LAYERS}"
set +x

echo "[$(date)] Done."
