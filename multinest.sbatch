#!/bin/bash
#SBATCH --job-name=kic5095_multinest
#SBATCH --output=kic5095.stdout
#SBATCH --error=kic5095.err
#SBATCH --time=60:00:00
#SBATCH --partition=fabrycky
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3000
#SBATCH --mail-type=FAIL,END

export LD_LIBRARY_PATH=/home/maxgoldberg/MultiNest/lib:$LD_LIBRARY_PATH
module load Anaconda3
module load hdf5
source activate ana
mpiexec python3 -u multinest.py 5095