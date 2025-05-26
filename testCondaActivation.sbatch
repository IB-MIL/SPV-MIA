#!/bin/bash
#SBATCH --job-name=conda_test
#SBATCH --output=conda_test_%j.out
#SBATCH --error=conda_test_%j.err
#SBATCH --partition=milan
#SBATCH --gres=
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:02:00

echo "Job Started: $(date)"
source /home/qhoat/miniconda3/etc/profile.d/conda.sh
conda activate llm-env
echo "Conda environment activated"
which python
python --version
