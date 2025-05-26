#!/bin/bash

#---------------------------------------------------------------------
# Slurm Directives - TARGETING Ngarrgu Tindebeek A100 GPU Nodes
#---------------------------------------------------------------------
#SBATCH --job-name=finetune_spr_gpt2xl_a100 
#SBATCH --output=finetune_spr_gpt2xl_%j.out 
#SBATCH --error=finetune_spr_gpt2xl_%j.err 
#SBATCH --partition=gpu-a100 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=00:05:00

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

# Load modules (adjust for OzSTAR)
echo "Loading modules..."
module purge
# module load anaconda3/YYYY.MM
# module load cuda/XX.Y
# module load cudnn/X.Y.Z

# Activate Conda environment 
echo "Activating Conda env: ${CONDA_ENV_NAME}"
source /fred/oz413/.conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME}"
if [ $? -ne 0 ]; then
    echo "ERROR: Could not activate Conda environment '${CONDA_ENV_NAME}'. Create it first."
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
# Execute Fine-Tuning with Accelerate
#---------------------------------------------------------------------
echo "Launching training with Accelerate..."

accelerate launch ./ft_llms/llms_finetune.py \
    --output_dir ./ft_llms/gpt2-xl/ag_news/baseline/ \
    --block_size 128 --eval_steps 100 --save_epochs 100 --log_steps 100 \
    -d ag_news -m gpt2-xl --packing --use_dataset_cache \
    -e 2 -b 4 -lr 5e-5 --gradient_accumulation_steps 1 \
    --train_sta_idx=0 --train_end_idx=10000 \
    --eval_sta_idx=0 --eval_end_idx=1000 \
    --dataset_config_name "default"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: llms_finetune.py failed with exit code $EXIT_CODE"
else
    echo "Fine-tuning completed successfully."
fi

#---------------------------------------------------------------------
# End of Job
#---------------------------------------------------------------------
echo "Job Ended: $(date)"
exit $EXIT_CODE