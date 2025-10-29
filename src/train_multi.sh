#!/usr/bin/env bash
#SBATCH --account=bdbq-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --constraint=scratch&projects
#SBATCH --job-name=egnn_multi
#SBATCH --output=slurm-egnn-%x-%j.out
#SBATCH --error=slurm-egnn-%x-%j.err

set -euo pipefail

# --- CONFIG ---
WORKSPACE="/scratch/bdbq/mcampos1/eigenEGNN_model"
PT_DIR="${WORKSPACE}/data_processed"
OUT_DIR="models"

EPOCHS=50
BATCH=4
LR=1e-3
LOSS="L2"
SCHED="Cos"
HIDDEN="64 128 256"
MLP_HID=64
MLP_LAYERS=2
AMP_FLAG="--amp"
VERBOSE=1

echo "[$(date)] Node: $(hostname)"
echo "CUDA visible: ${CUDA_VISIBLE_DEVICES:-unset}"

module reset
source /sw/external/python/anaconda3_gpu/bin/activate
conda activate /projects/bdbq/eigenvenv
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# --- locate train.py ---
cd "$SLURM_SUBMIT_DIR"
if [[ -f train.py ]]; then
  : # already here
elif [[ -d src && -f src/train.py ]]; then
  cd src
else
  echo "train.py not found"; exit 1
fi
echo "CWD: $(pwd)"

# --- helper function ---
run_dataset () {
  local DATA_NAME="$1"
  echo
  echo "=== Training dataset: ${DATA_NAME} ==="
  mkdir -p "${PT_DIR}"
  if [[ -f "${PT_DIR}/test.pt" && ! -f "${PT_DIR}/processed_${DATA_NAME}.pt" ]]; then
    echo "[info] Creating symlink processed_${DATA_NAME}.pt -> test.pt"
    ln -sfn "${PT_DIR}/test.pt" "${PT_DIR}/processed_${DATA_NAME}.pt"
  fi
  ls -lh "${PT_DIR}/processed_${DATA_NAME}.pt" || true

  python -u train.py \
    --workspace "${WORKSPACE}" \
    --output_dir "${OUT_DIR}" \
    --datasets "${DATA_NAME}" \
    --epochs "${EPOCHS}" \
    --batch "${BATCH}" \
    --lr "${LR}" \
    --loss "${LOSS}" \
    --scheduler "${SCHED}" \
    --hidden ${HIDDEN} \
    --mlp-hidden "${MLP_HID}" \
    --mlp-layers "${MLP_LAYERS}" \
    --workers 0 --eval-workers 0 --no-pin-memory \
    ${AMP_FLAG} \
    --verbose "${VERBOSE}"
}

# --- Run sequentially (1 GPU reused) ---
run_dataset "Tori"
run_dataset "ModelNet10"

echo "[$(date)] All datasets done."
