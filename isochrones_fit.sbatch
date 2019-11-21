#!/bin/bash
#SBATCH --job-name=kic5095_iso
#SBATCH --output=kic5095_iso.stdout
#SBATCH --error=kic5095_iso.err
#SBATCH --time=60:00:00
#SBATCH --partition=fabrycky
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3000
#SBATCH --mail-type=FAIL,END

export LD_LIBRARY_PATH=/home/maxgoldberg/MultiNest/lib:$LD_LIBRARY_PATH
module load Anaconda3
module load hdf5
source activate ana
python3 -u isochrones_fit.py