# sbatch utils/run_gnn_uiuc.sh
squeue -u $USER

## Running a job

Connecting to a cluster:
```bash
srun   --account=bdbq-delta-gpu   --partition=gpuA40x4-interactive   --nodes=1   --gpus-per-node=1   --ntasks=1   --cpus-per-task=1   --mem=20g   --time=01:00:00   --constraint="scratch&projects"   --pty /bin/bash
```

Load in the source files to use anaconda:
```bash
source /sw/external/python/anaconda3_gpu/bin/activate
```

Load the conda environment:
```
conda activate /projects/bdbq/eigenvenv
```