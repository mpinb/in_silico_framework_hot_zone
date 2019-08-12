#!/bin/bash -l
#SBATCH -p p.gpu # partition (queue)
#SBATCH -N 7 # number of nodes
#SBATCH -n 7 # number of cores
#SBATCH --mem MaxMemPerNode # 100000 # memory pool for all cores
#SBATCH -t 1-0:00 # time (D-HH:MM)
#SBATCH -o out.slurm.%N.%j.slurm # STDOUT
#SBATCH -e err.slurm.%N.%j.slurm # STDERR
#SBATCH --ntasks-per-node=20

srun -n7 -N7 -c20 python $MYBASEDIR/project_src/in_silico_framework/SLURM_scripts/component_1.py $MYBASEDIR/management_dir_$1
## sleep 3000
