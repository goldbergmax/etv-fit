#!/bin/bash
#SBATCH --job-name=kic5095_reduced_fit
#SBATCH --output=kic5095.stdout
#SBATCH --error=kic5095.err
#SBATCH --time=60:00:00
#SBATCH --partition=fabrycky
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3000
#SBATCH --mail-type=FAIL,END

module load Anaconda3
source activate ana
python3 -u im_g1_fit.py