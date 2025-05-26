#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=gpu_test.out

module purge
# module load cuda/XX.Y if needed
echo "Checking GPU..."
nvidia-smi
