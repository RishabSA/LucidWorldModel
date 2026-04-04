#!/bin/bash
#SBATCH --job-name=lwm
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm/%x_%j.out

# Runs one module on a GPU node. From the repo root:
# mkdir -p runs/slurm && sbatch -J train-snake sbatch/gpu.sbatch scripts.train --game snake --name base
set -euo pipefail

# uv lives in ~/.local/bin, which batch shells do not put on PATH
export PATH="$HOME/.local/bin:$PATH"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
uv run --frozen python -m "$@"
