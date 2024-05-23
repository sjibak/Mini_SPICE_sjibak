#!/bin/bash -l 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00 
#SBATCH --job-name=mpa_1
#SBATCH --export=NONE 

unset SLURM_EXPORT_ENV 

# cpus-per-task has to be set again for srun
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

#module load python/3.10-anaconda
source $HOME/spice/bin/activate
cd $HPCVAULT/Mini_SPICE
#ls
#python --version
srun python train.py
