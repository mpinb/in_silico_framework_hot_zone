#!/bin/bash -l
help() {
    cat <<EOF
  Usage: $0 {job name} [options]

  Arguments:

  name
    Name of the job
    Required

  -h, --help
    Display this usage message and exit.

  -N <val>
    Amount of nodes
    Default: 1

  -n <val>
    Amount of cores
    Default: 48

  -m <val>
    Amount of memory (in MB)
    default: 300,000 MB

  -t <val>
    Amount of time
    format: "d-h:m"

  -g
    Request a GPU partition.

  -i
    Launch an interactive job.

  -p <val>
    Request a specific partition. Overwrites the -i and -g flags.
    Default: "CPU"
    Options: [CPU, CPU-interactive, GPU, GPU-interactive, GPU-a100]

  -A
    Request resources on the A100 nodes.
    Overwrites the -g, -i and -p flags.

  -T <val>
    Amount of tasks per node
    Default: 20

  -r <val>
    Set value for gres.
    Default:
      1 for CPU jobs and non-interactive GPU jobs
      4 for interactive GPU jobs
EOF
}

usage() {
cat << EOF
  Usage: $0 {job name} [options]
EOF
}

qos_setting() {
  if [ $1 = "GPU-a100" ]
    echo "
    #SBATCH --qos=GPU-a100
    "
  fi
  echo ""
}

function args_precheck {
  if [ $1 -eq "0" ] ; then
    echo "Warning: no arguments passed. Will launch a job with default parameters and no name."
  elif [ $2 = "--help" ] || [ $2 = "-h" ] ; then
    help
    exit 0
  elif [[ $2 == -* ]] ; then
    echo "Warning: provided parameters before a job name. You must provide a job name first."
    usage
    exit 1
  fi
}


args_precheck $# $1;

# Default options
name=$1
shift;
nodes="1"
partition="CPU"
cores="48"
mem="300000"
time="1-0:00"
tpn="20"
gres="0"  # unset
a100=""  # unset
qos=""  # unset

# Parse options
while getopts hN:p:n:m:t:giA:T:r: OPT
do
    case "$OPT" in
        h) usage
           exit;;
        N) nodes=${OPTARG};;
        n) cores=${OPTARG};;
        m) mem=${OPTARG};;
	      t) time=${OPTARG};;
	      g) partition="GPU";;
	      i) partition=$p"-interactive";;  # append "-interactive"
        p) partition=${OPTARG};;  # overwrites i or g flag
        T) tpn=${OPTARG};;
        r) gres=${OPTARG};;
        A) a100="_"${OPTARG}; qos="GPU-a100";;  # appendix to start the correct python file
        \?) # incorrect option
         echo "Error: Invalid option"
         exit 1;;
        *) break;; # reached the list of file names
    esac
done

qosline = qos_setting $a100

shift $((OPTIND-1))  # just good practice

if [ $partition = "GPU-interactive" ] && [ $gres -eq "0" ]; then  # set gres to 4 for GPU-interactive jobs if it wasn't passed in command line
  gres = "4"
fi

# Print out job information
echo "
---------------------------------------------
Launching job named "$name" on $partition with
- $nodes nodes
- $cores cores 
- $mem memory
- for $time
"

# create wrapper script to start batch job with given parameters
sbatch << EoT
#!/bin/sh 
#SBATCH -p $partition # partition (queue)
#SBATCH -N $nodes # number of nodes
#SBATCH -n $cores # number of cores
#SBATCH --mem $mem # memory pool for all cores
#SBATCH -t $time # time (D-HH:MM)
#SBATCH -o out.slurm.%N.%j.slurm # STDOUT
#SBATCH -e err.slurm.%N.%j.slurm # STDERR
##SBATCH --ntasks-per-node=$tpn
##SBATCH --gres=gpu:$gres $qosline
#module load cuda
unset XDG_RUNTIME_DIR
export SLURM_CPU_BIND=none
ulimit -Sn "$(ulimit -Hn)"
srun -n1 -N$nodes -c$cores python $MYBASEDIR/project_src/in_silico_framework/SLURM_scripts/component_1_SOMA$a100.py $MYBASEDIR/management_dir_$name
EoT
echo "---------------------------------------------"