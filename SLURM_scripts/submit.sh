#!/bin/bash -l

# Default options
nodes="1"
partition="CPU"
cores="48"
mem="300000"
time="1-0:00"
tpn="20"
gres="1"  # default
qosline=""  # not set yet
cuda=""  # not set yet

help() {
  cat <<EOF
    Usage: $0 [options] {job name}

    Arguments:

    name
      Name of the job
      Required

    -h, --help
      Display this usage message and exit.

    -N <val>
      Amount of nodes
      Default: $nodes

    -n <val>
      Amount of cores
      Default: $cores

    -m <val>
      Amount of memory (in MB)
      default: $mem MB

    -t <val>
      Amount of time
      Format: "d-h:m"
      default: $time

    -g
      Request a GPU partition.

    -i
      Launch an interactive job.

    -p <val>
      Request a specific partition. Overwrites the -i and -g flags.
      Default: $partition
      Options: [CPU, CPU-interactive, GPU, GPU-interactive, GPU-a100]

    -A
      Request resources on the A100 nodes.
      Overwrites the -g, -i and -p flags.

    -T <val>
      Amount of tasks per node
      Default: $tpn

    -r <val>
      Set value for gres.
      Default:
        $gres for CPU jobs and non-interactive GPU jobs
        4 for interactive GPU jobs
EOF
}

usage() {
  cat << EOF
  Usage: $0 [options] {job name}
EOF
}

function args_precheck {
  if [ $1 -eq "0" ] ; then
    echo "Warning: no arguments passed. Will launch a job with default parameters and no name."
  elif [ $2 = "--help" ] || [ $2 = "-h" ] ; then
    help
    exit 0
  fi
}

args_precheck $# $1;

# Parse options
while getopts hN:p:n:m:t:A:T:r:gi OPT;
do
  case "$OPT" in
    h) usage
        exit;;
    N) nodes=${OPTARG};;
    n) cores=${OPTARG};;
    m) mem=${OPTARG};;
    t) time=${OPTARG};;
    g) partition="GPU";;
    i) partition=$partition"-interactive";;  # append "-interactive"
    p) partition=${OPTARG};;  # overwrites i or g flag
    T) tpn=${OPTARG};;
    r) gres=${OPTARG};;
    A) partition="GPU-a100";;  # appendix to start the correct python file
    \?) # incorrect option
      echo "Error: Invalid option"
      exit 1;;
  esac
done
shift $(( OPTIND - 1 ))  # shift the option index to point to the first non-option argument (should be name of the job)
name=$1
shift;
if [ -z "$name" ]
then
  echo "Warning: no jobname was passed. Job will start without a name."
fi



################### Cluster logic ###################
# Here, some extra variables are changed or created to add to the SLURM script depending on your needs

# If working on a GPU parition: load cuda (no matter how GPu partition was requested) with single hashtag, idk why?
if [ ${partition:0:3} = "GPU" ]; then
  cuda=$'\n#module load cuda'
fi

# Set gres to 4 if working on a GPU-interactive partition it if hasn't been set manually, load cuda
if [ $partition = "GPU-interactive" ] && [ $gres -eq "1" ]; then
  gres="4"
  cuda=$'\nmodule load cuda'
fi



# Manually set qos line and _A100 python file suffix in case the A100 was requested witht he -p flag instead of the A flag
if [ $partition = "GPU-a100" ]; then
  a100="_A100"  # python suffix to start the correct file
  qosline=$'\n#SBATCH --qos=GPU-a100'
fi




# TODO implement max possible mem or time

# Print out job information
echo "
---------------------------------------------
Launching job named \""$name"\" on $partition with
- $nodes nodes
- $cores cores 
- $mem MB memory
- gres: $gres
- for $time
"

# create wrapper script to start batch job with given parameters
# using EoF or EoT makes it easier to pass multi-line text with argument values
sbatch << EoF
#!/bin/bash -l
#SBATCH --job-name=$name
#SBATCH -p $partition # partition (queue)
#SBATCH -N $nodes # number of nodes
#SBATCH -n $cores # number of cores
#SBATCH --mem $mem # memory pool for all cores
#SBATCH -t $time # time (D-HH:MM)
#SBATCH -o out.slurm.%N.%j.slurm # STDOUT
#SBATCH -e err.slurm.%N.%j.slurm # STDERR
##SBATCH --ntasks-per-node=$tpn
#SBATCH --gres=gpu:$gres$qosline$cuda
unset XDG_RUNTIME_DIR
unset DISPLAY
export SLURM_CPU_BIND=none
ulimit -Sn "\$(ulimit -Hn)"
srun -n1 -N$nodes -c$cores python \$MYBASEDIR/project_src/in_silico_framework/SLURM_scripts/component_1_SOMA$a100.py \$MYBASEDIR/management_dir_$name
## sleep 3000
EoF
echo "---------------------------------------------"
