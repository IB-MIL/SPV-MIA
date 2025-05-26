#!/bin/bash

#---------------------------------------------------------------------
# Slurm Directives - Targeting Available GPU Partition (milan-gpu)
#---------------------------------------------------------------------
#SBATCH --job-name=attack_python_gpt2xl
#SBATCH --output=attack_python_ibrahim_%j.out
#SBATCH --error=attack_python_ibrahim_%j.err
#SBATCH --partition=milan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

#---------------------------------------------------------------------
# USER CONFIGURATION
#---------------------------------------------------------------------
BASE_PROJECT_DIR="/fred/oz413/ANeurIPS2024_SPV-MIA"
CONDA_ENV_NAME="llm-env"

#---------------------------------------------------------------------
# Environment Setup
#---------------------------------------------------------------------
echo "Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node(s): $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated Memory: $SLURM_MEM_PER_NODE MB"
echo "Allocated GPU(s): $SLURM_JOB_GPUS (using GRES: $SLURM_GRES)"

# Load modules (adjust if needed)
echo "Loading modules..."
module purge
# module load anaconda3/YYYY.MM
# module load cuda/XX.Y
# module load cudnn/X.Y.Z

# Activate Conda environment
echo "Activating Conda env: ${CONDA_ENV_NAME}"
source /home/ibarmil/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME}"
if [ $? -ne 0 ]; then
    echo "ERROR: Could not activate Conda environment '${CONDA_ENV_NAME}'."
    exit 1
fi
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

# Change to the base project directory
cd "${BASE_PROJECT_DIR}" || {
    echo "ERROR: Failed to cd into ${BASE_PROJECT_DIR}"
    exit 1
}
echo "Working directory: $(pwd)"

#---------------------------------------------------------------------
# Execute Attack Script
#---------------------------------------------------------------------
echo "Running attack.py..."
python attack_ibrahim.py

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: attack.py failed with exit code $EXIT_CODE"
else
    echo "Attack script completed successfully."
fi

#---------------------------------------------------------------------
# End of Job
#---------------------------------------------------------------------
echo "Job Ended: $(date)"
exit $EXIT_CODE
