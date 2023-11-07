#!/bin/bash -l

# these variables allow the main shell to exit from subshells, which is used in functions (since they create subshells)
# If some nested subshell kills the main shell, its direct parent will keep running.
trap "exit 1" SIGUSR1
PROC=$$

# If a job doesnt start, this string will change tot he reason as to why it doesnt
QOS=""

# The user-defined port numbers of jupyter notebook/lab
__LOCATION__="$(dirname "$(realpath "$0")")"
DASK_PORT=$(awk -F "=" '/dask_client_3/ {print $2}' $__LOCATION__/user_settings.ini)
NOTEBOOK_PORT=$(awk -F "=" '/jupyter_notebook/ {print $2}' $__LOCATION__/user_settings.ini)
LAB_PORT=$(awk -F "=" '/jupyter_lab/ {print $2}' $__LOCATION__/user_settings.ini)

NC='\033[0m' # No Color
ORANGE='\033[0;33m' # orange color

#######################################
# Given a string, continuously prints the string with an 
# updated "..." icon
# Arguments:
#   1. A string
#######################################
# Little spinning icon
dot[0]="   "
dot[1]=".  "
dot[2]=".. "
dot[3]="..."
function printf_with_dots {
    local string=$1
    local width="$(tput cols)"
    local string_length=${#string}
    local padding=$(($width - $string_length - 4))  # to clean out previous long strings
    for i in "${dot[@]}"
    do
        # \r removes previous line
        # \b$i is the ... icon
        # >&2 writes to stdout
        printf "\r$1\b$i%*s" "$padding">&2
        sleep 0.3
    done
}

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
    kill -SIGUSR1 $PROC; exit 1  # kill main shell and exit function
  fi
}

#######################################
# Search through the management_dir for the scheduler file that contains the IP of the running job
# Arguments:
#   1: The name of the job
#######################################
function fetch_ip_from_management_dir {
    local sfile="$MYBASEDIR/management_dir_$1/scheduler3.json"
    while ! test -f $sfile; do
        printf_with_dots "Waiting for creation of $sfile "
    done;
    # Search through sfile output. Grep for http://<ip>:<dask_port> and ?/token=<token> (if a token exists)
    # Keep searching as long as it does not appear in output
    local scheduler_address=$(cat $sfile | grep -Eo -m1 "tcp://[0-9]{2,3}\.[0-9]+\.[0-9]+\.[0-9]+:$DASK_PORT" )
    while [ -z "$scheduler_address" ]; do  # wait until link is written and grep returns a match
        printf_with_dots "Launching Jupyter Lab server "
        local scheduler_address=$(cat $sfile | grep -Eo -m1 "tcp://[0-9]{2,3}\.[0-9]+\.[0-9]+\.[0-9]+:$DASK_PORT" )
    done;
    # remove prefix and suffix to get just the ip
    ip=${scheduler_address#"tcp://"}
    ip=${ip%":$DASK_PORT"}
    echo "${ip}"
}


#######################################
# Search through the management_dir jupyter.txt file for the jupyter link (including token if it is configured to use it)
#   greps for 11113 (the default port for jupyter lab) and outputs the first match.
# Arguments:
#   1: The name of the job
#   2: Which port to search for (11112 for notebook, 11113 for lab)
#######################################
function fetch_jupyter_link_from_file {
    local jupyter_file="$MYBASEDIR/management_dir_$1/jupyter.txt"
    while ! test -f $jupyter_file; do
        printf_with_dots "Setting up jupyter lab server "
    done;
    # Search through jupyter output. Grep for http://<ip>:<port> and ?/token=<token> (if a token exists)
    # Keep searching as long as it does not appear in output
    local link=$(cat $jupyter_file | grep -Eo -m1 "http://[0-9]{2,3}\.[0-9]+\.[0-9]+\.[0-9]+:$2[/?token=[a-z0-9]*]?" )
    while [ -z "$link" ]; do  # wait until link is written and grep returns a match
        printf_with_dots "Launching Jupyter Lab server "
        local link=$(cat $jupyter_file | grep -Eo -m1 "http://[0-9]{2,3}\.[0-9]+\.[0-9]+\.[0-9]+:$2[/?token=[a-z0-9]*]?" )
    done;
    echo "${link}"
}

#######################################
# Jupyter outputs the link with an IP different from the one we want to use.
# This function replaces the IP with the correct one.
# Arguments:
#   1: The name of the job
#   2: The jupyter link with wrong IP
#   3: The port of the jupyter server (11112 for notebook, 11113 for lab)
#######################################
function format_jupyter_links {
    local ip=$(fetch_ip_from_management_dir $1)
    local token=$(echo $2 | grep -Eo "(/\?token=[a-z0-9]*)?")
    echo http://$ip:$3$token
}

# Main
args_precheck $#;  # check amount of arguments
job_name="$1"
jupyter_file="$MYBASEDIR/management_dir_$job_name/jupyter.txt"
notebook_link="$(fetch_jupyter_link_from_file $job_name $NOTEBOOK_PORT)";
lab_link="$(fetch_jupyter_link_from_file $job_name $LAB_PORT)";
# insert IP of correct node in jupyter link
notebook_link=$(format_jupyter_links $job_name $notebook_link $NOTEBOOK_PORT);
lab_link=$(format_jupyter_links $job_name $lab_link $LAB_PORT);

width="$(tput cols)"
printf '\r%*s' $width  # clear previous line
printf "${ORANGE}Jupyter Lab${NC} server for \"$1\" is running at:
$lab_link

${ORANGE}Jupyter notebook${NC} server for \"$1\" is running at:
$notebook_link
\n"
exit 0;
