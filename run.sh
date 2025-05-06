#!/bin/bash

#SBATCH -A che190010      # Ensure correct allocation
#SBATCH --nodes=1             # Total # of nodes 
#SBATCH --ntasks-per-node=1   # Number of MPI ranks per node (one rank per GPU)    
#SBATCH --time=9:30:00
#SBATCH -J Be
#SBATCH -o logs/SVD-tenth-rows.out
#SBATCH -p wholenode  # Ensure partition exists


# Load necessary modules
module purge
module load gcc/11.2.0  # Load a compatible compiler
module load openmpi/4.0.6  # Now OpenMPI should load correctly

# Activate Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fitsnap

# Run the job using SLURM's srun (recommended over mpirun)
mpirun -n 1 python qACE-random.py
