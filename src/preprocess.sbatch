#!/usr/bin/env bash
#SBATCH --account=bdbq-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --constraint=scratch&projects
#SBATCH --job-name=pre_eigs
#SBATCH --output=slurm-pre-%x-%A_%a.out
#SBATCH --error=slurm-pre-%x-%A_%a.err
#SBATCH --array=0-3

set -euo pipefail

# Datasets to preprocess (array index selects one)
DATASETS=(ModelNet10 SynthNet Thingi10k)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

NUM_EIGS=10
LIMIT=-1            # -1 = no cap
ALLOW_MISSING=""    # set to "--allow-missing" if you want to keep graphs with missing eigs
TEST_FLAG=""        # set to "--test" for batch summaries

echo "[$(date)] Node: $(hostname)"
echo "Task ${SLURM_ARRAY_TASK_ID} -> dataset=${DS}"

module reset
source /sw/external/python/anaconda3_gpu/bin/activate
conda activate /projects/bdbq/eigenvenv

# --- Find repo root and cd so that preprocessing.py is found ---
# If this sbatch file is in src/, SLURM_SUBMIT_DIR already points to .../eigenEGNN_model/src
cd "$SLURM_SUBMIT_DIR"

# If user accidentally submits from repo root and sbatch file lives in src/, this still works:
if [[ ! -f preprocessing.py && -d src && -f src/preprocessing.py ]]; then
  cd src
fi

pwd
ls -l preprocessing.py || { echo "preprocessing.py not found in $(pwd)"; exit 1; }

echo "Running: python preprocessing.py --data ${DS} --num-eigs ${NUM_EIGS} --limit ${LIMIT} ${ALLOW_MISSING} ${TEST_FLAG}"
python preprocessing.py --data "${DS}" --num-eigs "${NUM_EIGS}" --limit "${LIMIT}" ${ALLOW_MISSING} ${TEST_FLAG}

echo "[$(date)] Done dataset=${DS}"
