#!/bin/bash -l

# these variables allow the main shell to exit from subshells, which is used in functions
trap "exit 1" 10
PROC="$$"

# Assumes a job has been started with name <name> and its corresponding err, out files and management directory.

#######################################
# A quick check to see if the input arguments are what you would expect
#   only checks the amount of arguments for now
# Arguments:
#   1. Amount of arguments
#######################################
function args_precheck {
  if [ $1 -eq "0" ] ;
  then
    echo "Warning: no arguments passed. Please pass the name of the job as a parameter." >&2  # print error to stderr
    kill -10 $PROC;
  fi
}

#######################################
# Request the ID of some job name
#    Shows all running jobs of user, grep for jobname (truncated to first 8 chars), 
#    grep for 7 digit Job-ID with regex (-E option), return only regex match (-o option)
#    instead of full line
# Arguments:
#   1: The name of the job
#######################################
function fetch_id {
    local id="$(squeue -u $USER | grep ${1:0:9} | grep -Eo [0-9]{7})"
    if [ -z "$id" ] ; # check if node is found
    then
        echo "No JobID found for job name \"$1\" your SLURM queue.
Is the job name spelled correctly?" >&2  # print error to stderr
        kill -10 $PROC;
    else
        echo "${id}"
    fi
    
}

#######################################
# Requests the name of the node some job is running on
#   Show all running jobs of user, grep for jobname (truncated to first 8 chars), 
#   grep for somegpu or somacpu with three numbers at the end
#   TODO: add a100 regex
# Arguments:
#   1: The name of the job
#######################################
function fetch_node_name {
    local node="$(squeue -u $USER | grep ${1:0:8} | grep -Eo soma[cg]pu[0-9]{3})"
    if [ -z "$node" ] ; # check if node is found
    then
        echo "No node found for job name \"$1\" in your SLURM queue.
It may still be waiting to be assigned, or the job name may be spelled incorrectly." >&2  # print error to stderr
        kill -10 $PROC;
    else
        echo "${node}"
    fi
}

#######################################
# Search through the slurm err file for the jupyter link (including token if it is configured to use it)
#   greps for 11113 (the default port for jupyter lab) and outputs the first match
# Arguments:
#   1: The name of the node (e.g. somacpu089)
#   2: The ID of the job (7-number digit)
#######################################
function fetch_link {
    if [ -f "err.slurm.$1.$2.slurm" ] ; # check if err file exists
    then
        local link="$(cat err.slurm.$1.$2.slurm | grep -Eo http://$1:11113.* | head -1)"
        if [ -z "$link" ]  # check if node is found
        then
            echo "No Jupyter link found (yet) in \"err.slurm.$1.$2.slurm\"
The server has not been started (yet). Check if the job is running correctly." >&2  # print error to stderr
            kill -10 $PROC;
        else
            echo "${link}"
        fi
    else
        echo "File \"err.slurm.$1.$2.slurm\" does not exist (yet).
Check if the job is running correctly, or wait until the file has been created." >&2  # print error to stderr
        kill -10 $PROC;
    fi
}


#######################################
# Search through the management_dir scheduler.json file for the IP of the node
#   greps for tcp:// and 28786 (the default port for python 2 scheduler) and extracts the ip from inbetween
# Arguments:
#   1: The name of the job
#######################################
function fetch_ip {
    if [ ! -f "management_dir_$1/scheduler.json" ] ; # check if management_dir_*/scheduler.json file exists
    then
        echo "File management_dir_$1/scheduler.json does not exist(yet).
Check if the job name is spelled correctly and the job is running.
If this is the case, wait until the file has been created." >&2  # print error to stderr
        kill -10 $PROC;
    else
        local ip="$(cat management_dir_$1/scheduler.json | grep -Eo tcp://\.*28786 | grep -o -P '(?<=tcp://).*(?=:28786)')"
        echo "${ip}"
    fi
}

#######################################
# Construct the Jupter server link based on the node IP adress and the jupyter link shown in the output err file
#   Should work whether or not you have a token configured (only tested on token-having links)
# Arguments:
#   1: The name of the job
#   2: The name of the node
#   3: the id of the job
#######################################
function fetch_jupyter_link {
    local ip="$(fetch_ip $1)"
    local link="$(fetch_link $2 $3)"
    local link_suffix="$(echo $link | grep -oP '(?<=:11113/)'.* | head -1)"  # grep for anything after the port number
    if [ -z "$link_suffix" ] ;  # no suffix is found: no token, but also no 
    then
        echo "
Warning: No token is set for this Jupyter server.
VSCode will not be able to connect to this server, and notebooks running on this server can not be shared by link">&2
    fi
    local jupyter_link="http://$ip:11113/$link_suffix"
    echo $jupyter_link
}

args_precheck $#;  # check amount of arguments
job_name="$1"
id="$(fetch_id $1)"
node_name="$(fetch_node_name $1)"
link="$(fetch_link $node_name $id)"
jupyter_link="$(fetch_jupyter_link $job_name $node_name $id)";

echo "
Found job name \"$1\" with ID $id on $node_name

Jupyter lab is running at:
$jupyter_link
"
exit 0;