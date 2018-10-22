#!/bin/bash
#SBATCH --job-name=kic5095_py
#SBATCH --output=kic5095.stdout
#SBATCH --error=kic5095.err
#SBATCH --time=60:00:00
#SBATCH --partition=fabrycky
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=3000
#SBATCH --mail-type=FAIL,END

module load Anaconda3
source activate ana
python3 mcmc.py 5095 10000