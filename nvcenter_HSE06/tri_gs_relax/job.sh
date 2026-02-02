#!/bin/bash
#SBATCH -J hse
#SBATCH -o python.err
#SBATCH -p wzhcnormal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=48:00:00
module purge
module load compiler/gcc/9.3.0           mpi/openmpi/4.1.5/gcc-9.3.0
#export PATH=/work/home/ac2uoma125/anaconda3/bin:$PATH
source activate gpaw2460
#conda activate gpaw2460

mpirun -n 32 gpaw python run.py

