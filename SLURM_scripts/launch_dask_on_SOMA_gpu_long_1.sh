#!/bin/bash -l
#SBATCH -p CPU-long # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -n 48 # number of cores
#SBATCH --mem 300000 # memory pool for all cores
#SBATCH -t 5-0:00 # time (D-HH:MM)
#SBATCH -o out.slurm.%N.%j.slurm # STDOUT
#SBATCH -e err.slurm.%N.%j.slurm # STDERR
##SBATCH --ntasks-per-node=20
##SBATCH --gres=gpu:1
#module load cuda
unset XDG_RUNTIME_DIR
export SLURM_CPU_BIND=none
ulimit -Sn "$(ulimit -Hn)"
srun -n1 -N1 -c48 python $MYBASEDIR/project_src/in_silico_framework/SLURM_scripts/component_1_SOMA.py $MYBASEDIR/management_dir_$1

