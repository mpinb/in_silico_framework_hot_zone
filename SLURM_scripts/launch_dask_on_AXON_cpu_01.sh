#!/bin/bash -l
#SBATCH -p p.axon # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH --mem 100000 # memory pool for all cores
#SBATCH -t 1-0:00 # time (D-HH:MM)
#SBATCH -o out.slurm.%N.%j.slurm # STDOUT
#SBATCH -e err.slurm.%N.%j.slurm # STDERR
#SBATCH --ntasks-per-node=40

srun -n1 -N1 -c40 python $MYBASEDIR/project_src/in_silico_framework/SLURM_scripts/component_1.py $MYBASEDIR/management_dir_$RANDOM
sleep 3000
