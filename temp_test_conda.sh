#!/bin/bash
#SBATCH --job-name=test_conda
#SBATCH --output=test_conda_%j.out
#SBATCH --error=test_conda_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00

source /fred/oz413/.conda/etc/profile.d/conda.sh
conda activate /fred/oz413/.conda/envs/llm-env

echo "Conda env path: $CONDA_PREFIX"
which python
python --version
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
