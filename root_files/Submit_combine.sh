#!/bin/bash
#SBATCH --job-name=combine
#SBATCH --output=combine.out
#SBATCH --error=combine.err
#SBATCH --time=00:20:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1

echo "Running combine_files.sh"
bash Combine_files.sh