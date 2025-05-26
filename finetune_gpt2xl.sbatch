#!/bin/bash

#---------------------------------------------------------------------
# Slurm Directives - OzSTAR (No Partition Specified)
#---------------------------------------------------------------------
#SBATCH --job-name=finetune_spr_gpt2xl_auto_part
#SBATCH --output=finetune_spr_gpt2xl_%j.out
#SBATCH --error=finetune_spr_gpt2xl_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8              # Number of CPU cores 
#SBATCH --mem=64G                      # Total CPU memory for the job
#SBATCH --gres=gpu:1                   # Request GPU. 
                                       
#SBATCH --time=01:00:00                

#---------------------------------------------------------------------
# USER CONFIGURATION
#---------------------------------------------------------------------
BASE_PROJECT_DIR="/fred/oz413/ANeurIPS2024_SPV-MIA"
CONDA_ENV_NAME="llm-env"
export WANDB_MODE="disabled"
#---------------------------------------------------------------------
# Environment Setup
#---------------------------------------------------------------------
echo "Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node(s): $SLURM_NODELIST"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated Memory: $SLURM_MEM_PER_NODE MB"
echo "Allocated GPU(s) Raw: $SLURM_JOB_GPUS (using GRES: $SLURM_GRES)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo "Attempting to load system Mamba module..."
module purge
module load mamba

if ! command -v conda &> /dev/null
then
    echo "ERROR: conda command not found. Check mamba module load."
    exit 1
fi
echo "Conda (mamba) command found at: $(which conda)"

echo "Activating Conda env: ${CONDA_ENV_NAME}"
conda activate "${CONDA_ENV_NAME}"
if [ $? -ne 0 ]; then
    echo "ERROR: Could not activate Conda environment '${CONDA_ENV_NAME}'. Ensure it exists and Conda/Mamba is initialized correctly."
    exit 1
fi

echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'PyTorch not found or error in import')"
echo "Number of GPUs PyTorch sees: $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 'PyTorch not found or error in import')"
nvidia-smi || echo "nvidia-smi not found or failed"

cd "${BASE_PROJECT_DIR}" || {
    echo "ERROR: Failed to cd into ${BASE_PROJECT_DIR}"
    exit 1
}
echo "Working directory: $(pwd)"

#---------------------------------------------------------------------
# Execute Fine-Tuning with Accelerate
#---------------------------------------------------------------------
echo "Launching training with Accelerate..."
# CONFIGURE `accelerate config` for best results,
# or explicitly pass all necessary arguments to accelerate launch.
accelerate launch \
    --mixed_precision fp16 \
    --num_processes 1 \
    --num_machines 1 \
    ./ft_llms/llms_finetune.py \
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
