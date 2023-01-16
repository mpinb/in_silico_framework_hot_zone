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
launch_jupyter_server="0"

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
    echo "Warning: no arguments passed. Will launch a job with default parameters and no name."
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
      echo "A job with this name is already running. Consider renaming the job and resubmitting."
      exit 1
    fi
    return 0
}

#######################################
# Requests the name of the node some job is running on
#   Show all running jobs of user, grep for jobname (truncated to first 8 chars), 
#   grep for somegpu or somacpu with three numbers at the end
#   TODO: add a100 regex
# Arguments:
#   1: The ID fo the job
#######################################
function fetch_node_name {
    local node_name="$(squeue -u $USER | grep $1 | grep -Eo soma[cg]pu[0-9]{3})"
    echo "${node_name}"
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
    i) partition=$partition"-interactive";launch_jupyter_server="1";;  # append "-interactive"
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

if [ $partition = "GPU-interactive" ]; then
  cuda=$'\nmodule load cuda'
  if [ $gres -eq "0" ]; then  # Set gres to 4 if working on a GPU-interactive partition it if hasn't been set manually
    gres="4"
  fi
elif [ ${partition:0:3} == CPU ]; then
  cuda=$'\nmodule load cuda'
  gres="0"
fi

# Manually set qos line and _A100 python file suffix in case the A100 was requested with the -p flag instead of the A flag
if [ $partition = "GPU-a100" ]; then
  a100="_A100"  # python suffix to start the correct file
  qosline=$'\n#SBATCH --qos=GPU-a100'
  if [ $cores \> 32 ]; then
    cores=32
  fi
fi

if [ ! -z "$tpn" ]; then
  tpn="#SBATCH --ntasks-per-node="$tpn
fi



# TODO implement max possible mem or time

# Print out job information
echo "
---------------------------------------------
Launching job named \""$name"\" on $partition with
- $nodes nodes
- $cores cores 
- $memstr MB memory
- gres: $gres
- for $time
"

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
srun -n1 -N$nodes -c$cores python -u \$MYBASEDIR/project_src/in_silico_framework/etc/SLURM_scripts/component_1_SOMA.py \$MYBASEDIR/management_dir_$name $launch_jupyter_server
EoF
)
echo $output
echo "---------------------------------------------"
id="$(echo $output | grep -Eo [0-9]{7})"  # grep slurm submit output for ID


# ##### Fetching the jupyter link from the err file
# # continuously monitor cwd until the error file exsists
# echo "Waiting for err.slurm file to exist"
# while [ ! $(ls err.slurm.*.$id.slurm 2> /dev/null) ]
# do
#   sleep 0.1
# done
# # Find name of the requested node
# node_name="$(fetch_node_name $id)"
# if [ -z "$node_name" ]; then
#   echo "No node found for job ID $id. Has it been assigned yet?"
#   exit 1
# fi
# echo "Found err.slurm.$node_name.$id.slurm"
# echo "Monitoring err.slurm.$node_name.$id.slurm for a Jupyter link..."
# # Continuously monitor err slurm file, when the grep finds something that looks like
# # a jupyter link starting with a number (i.e. the IP), stops monitoring (& command) and saves to variable
# # link="$( (tail -F err.slurm.$node_name.$id.slurm &) | grep -qEo http://[0-9].*:11113.* )"
# #TODO this doesnt work yet, it just keeps searching and doesnt match the grepe, or the grep doesnt stop the tail -f commmand idk
# link="$( tail -f -n0 logfile.log & ) | grep http://[0-9].*:11113.* "

# echo $link

# if [ -z "$link" ];  # check if a jupyter link is present in the err log
# then
#     echo "No jupyter link found in err.slurm.$node_name.$id.slurm
#     Check if the compute node is running properly"
#     exit 1;
# else
#     echo "
#     Found job name \"$1\" with ID $(fetch_id $1) on $(fetch_node_name $1)
    
#     Jupyter lab is running at:
#     $link
#     "
# fi
# exit 0;