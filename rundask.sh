#!/bin/bash

#SBATCH -A che190010      # Ensure correct allocation
#SBATCH --job-name=dask-tsqr-job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -p wholenode  # Ensure partition exists
#SBATCH --time=03:00:00
#SBATCH --output=logs/SVD-tenth-rows.o%j
#SBATCH --exclusive

# Load necessary modules
module purge
module load gcc/11.2.0  # Load a compatible compiler
module load openmpi/4.0.6  # Now OpenMPI should load correctly

# Activate Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fitsnap

python -u qACE-dask.py

