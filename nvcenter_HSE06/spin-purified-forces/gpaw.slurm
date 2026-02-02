#!/bin/bash
#SBATCH -J hse
#SBATCH -p wzhcnormal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4

module purge
module load compiler/gcc/9.3.0           mpi/openmpi/4.1.5/gcc-9.3.0
#export PATH=/work/home/ac2uoma125/anaconda3/bin:$PATH
source activate gpaw2460
#conda activate gpaw2460

mpirun -n 64 gpaw python relax.py

