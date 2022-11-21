#!/bin/bash -l

# Assumes a job has been started with name <name> and its corresponding err, out files and management directory.

function fetch_id {
    # Show all running jobs of user, grep for jobname, grep for 7 digit Job-ID with regex (-E option), return only regex match (-o option)
    # instead of full line
    id="$(squeue -u $USER | grep $1 | grep -Eo [[:digit:]]{7})"
    echo "${id}"
}

function fetch_node_name {
    # Show all running jobs of user, grep for jobname, grep for somegpu or somacpu with three numbers at the end
    # TODO: add a100 regex
    id="$(squeue -u $USER | grep $1 | grep -Eo soma[cg]pu[[:digit:]]{3})"
    echo "${id}"
}

function args_precheck {
  if [ $1 -eq "0" ] ; then
    echo "Warning: no arguments passed. Please pass the name of the job as a parameter."
    exit 1
  fi
}

#######################################
# Search through the slurm err file for the jupyter link (including token if it is configured to use it)
# greps for 11113 (the default port for jupyter lab) and outputs the first match
# Arguments:
#   1: The name of the node (e.g. somacpu089)
#   2: The ID of the job (7-number digit)
#######################################
function fetch_link {
    link="$(cat err.slurm.$1.$2.slurm | grep 11113 | grep -Eo http://$1:11113.* | head -1)"
    echo "${link}"
}

#######################################
# Search through the management_dir scheduler.json file for the IP of the node
# greps for tcp:// and 28786 (the default port for python 2 scheduler) and extracts the ip from inbetween
# Arguments:
#   1: The name of the job
#######################################
function fetch_ip {
    ip="$(cat management_dir_$1/scheduler.json | grep -Eo tcp://\.*28786 | grep -o -P '(?<=tcp://).*(?=:28786)')"
    echo "${ip}"
}

#######################################
# Construct the Jupter server link based on the node IP adress and the jupyter link shown in the output err file
# Should work whether or not you have a token configured (only tested on token-having links)
# Arguments:
#   1: The name of the job
#   2: The name of the node
#   3: the id of the node
#######################################
function fetch_jupyter_link {
    ip="$(fetch_ip $1)"
    link="$(fetch_link $2 $3)"
    link_suffix="$(echo $link | grep -o -P '(?<=:11113).*')"
    jupyter_link="http://$ip:11113$link_suffix"
    echo "${jupyter_link}"
}

args_precheck $#;

job_name="$1"
id="$(fetch_id $1)"
node_name="$(fetch_node_name $1)"

echo "
Found job name \"$1\" with ID $(fetch_id $1) on $(fetch_node_name $1)

Jupyter lab is running at:
$(fetch_jupyter_link $1 $node_name $id)
"
