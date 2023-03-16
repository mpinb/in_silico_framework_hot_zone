#!/bin/bash -l

# Default options
nodes="1"
partition="GPU"
cores="48"
mem="0"
memstr="max"
time="1-0:00"
tpn=""
gres="4"  # default for GPU jobs
qosline=""  # not set yet
cuda=$'\n#module load cuda'  # If working on a GPU partition: load cuda with single hashtag, idk why?
interactive="0"

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

    -c
      Request a CPU partition.

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
      Default: Not used

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
    echo "Warning: no arguments passed. Launching a job with default parameters and no name."
  elif [ $2 = "--help" ] || [ $2 = "-h" ] ; then
    help
    exit 0
  fi
}

#######################################
# Checks if a job is already running with same name
# Arguments:
#   1: The name of the job
#######################################
function check_name {
    local n_matches="$(squeue -u $USER | grep ${1:0:8} | wc -l)"
    if [ $n_matches -gt 0 ]; then
      echo "A job with this name is already running. Requested resources will be appended to job \"$1\""
    fi
    return 0
}

args_precheck $# $1;

# Parse options
while getopts "hN:n:m:t:cip:T:r:A" OPT;
do
  case "$OPT" in
    h) usage
        exit;;
    N) nodes=${OPTARG};;
    n) cores=${OPTARG};;
    m) mem=${OPTARG};memstr=$mem;;
    t) time=${OPTARG};;
    c) partition="CPU";;
    i) interactive="1";;
    p) partition=${OPTARG};;  # overwrites i or c flag
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
check_name $name

################### Cluster logic ###################
# Here, some extra variables are changed or created to add to the SLURM script depending on your needs

# Adapt partition name for interactive jobs
if [ ${#partition} == 3 -a $interactive == "1" ]; then
  # GPU or CPU interactive session
  # adapt partition name to have -interactive in it
  partition=$partition"-interactive"
fi

if [[ ${partition:0:3} == CPU ]]; then
    gres="0"
fi

# Loading cuda and setting gres depending on GPU/CPU partition
if [ $interactive == 1 ]; then
  interactive_string=" interactive"
  cuda=$'\nmodule load cuda'
  if [ ${partition:0:3} != "CPU" -a $gres -eq 0 ]; then
    # Either GPU-interactive or A100 interactive
    # Set gres to 4 if working on a GPU-interactive partition it if hasn't been set manually already
    gres="4"
  fi
elif [ $interactive == 0 ]; then
  interactive_string=""
fi

# Manually set qos line in case the A100 was requested with the -p flag instead of the A flag
if [ $partition = "GPU-a100" ]; then
  qosline=$'\n#SBATCH --qos=GPU-a100'
  if [ $cores \> 32 ]; then
    cores=32
  fi
fi

if [ ! -z "$tpn" ]; then
  tpn="#SBATCH --ntasks-per-node="$tpn
fi

################### Print out job info ###################
width="$(tput cols)"
printf '%0.s-' $(seq 1 $width)  # fill width with "-" char
echo "
Launching$interactive_string job named \""$name"\" on $partition with
- $nodes nodes
- $cores cores 
- $memstr MB memory
- gres: $gres
- for $time
"

################### Run SLURM scripts ###################
# create wrapper script to start batch job with given parameters
# using EoF or EoT makes it easier to pass multi-line text with argument values
# This submits the job and catches the output
output=$(sbatch <<EoF
#!/bin/bash -l
#SBATCH --job-name=$name
#SBATCH -p $partition # partition (queue)
#SBATCH -N $nodes # number of nodes
#SBATCH -n $cores # number of cores
#SBATCH --mem $mem # memory pool for all cores
#SBATCH -t $time # time (D-HH:MM)
#SBATCH -o out.slurm.%N.%j.slurm # STDOUT
#SBATCH -e err.slurm.%N.%j.slurm # STDERR
#SBATCH --gres=gpu:$gres$qosline$cuda
$tpn
unset XDG_RUNTIME_DIR
unset DISPLAY
export SLURM_CPU_BIND=none
ulimit -Sn "\$(ulimit -Hn)"
srun -n1 -N$nodes -c$cores python -u \$MYBASEDIR/project_src/in_silico_framework/SLURM_scripts/component_1_SOMA.py \$MYBASEDIR/management_dir_$name $interactive
EoF
)
echo $output
printf "Use squeue to check its running status\n"
printf '%0.s-' $(seq 1 $width)  # fill width with "-" char
echo ""
if [ $interactive == "1" ]; then
  # fetch working directory of current script
  __dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  # also run jupyter_link to fetch the link where the jupyter server is running
  bash ${__dir}/jupyter_link.sh $name
else
  printf "Batch job: no jupyter server will be launched.\n"
fi