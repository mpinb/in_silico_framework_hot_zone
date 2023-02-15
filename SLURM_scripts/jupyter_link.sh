#!/bin/bash -l

# these variables allow the main shell to exit from subshells, which is used in functions (since they create subshells)
# If some nested subshell kills the main shell, its direct parent will keep running.
trap "exit 1" SIGUSR1
PROC=$$


#######################################
# Given a string, continuously prints the string with an 
# updated spinner icon
# Arguments:
#   1. A string
#######################################
# Little spinning icon
spin[0]="-"
spin[1]="\\"
spin[2]="|"
spin[3]="/"
function printf_with_spinner {
    local string=$1
    local width="$(tput cols)"
    local string_length=${#string}
    local remainder=$(($width - $string_length - 1))
    for i in "${spin[@]}"
    do
        # \r removes previous line
        # %*s adds whitespaces for padding
        # amount of padding is given as an argument, hence the * char
        # \b$i is the spinner icon
        # >&2 writes to stdout
        printf "\r$1%*s\b$i" "$remainder">&2
        sleep 0.1
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
# Search through the management_dir jupyter.txt file for the jupyter link (including token if it is configured to use it)
#   greps for 11113 (the default port for jupyter lab) and outputs the first match.
# Arguments:
#   1: The name of the job
#######################################
function fetch_jupyter_link {
    local jupyter_file="$MYBASEDIR/management_dir_$1/jupyter.txt"
    while ! test -f $jupyter_file; do
        printf_with_spinner "Launching Jupyter server"
    done
    local link="$(cat $jupyter_file | grep -Eo http://.*:11113/.* | head -1)"
    while [ -z "$link" ]; do  # wait until link is written and grep returns a match
        printf_with_spinner "Launching Jupyter server"
        local link="$(cat $jupyter_file | grep -Eo http://.*:11113/.* | head -1)"
    done
    echo "${link}"
}


#######################################
# Search through the management_dir scheduler.json file for the IP of the node
#   greps for tcp:// and 28786 (the default port for python 2 scheduler) and extracts the ip from inbetween
# Arguments:
#   1: The name of the job
#######################################
function fetch_ip {
    management_dir="$MYBASEDIR/management_dir_$1"
    while ! test -d "$management_dir"; do
        printf_with_spinner "Creating \"management_dir_$1\" "
    done
    while ! test -f "$management_dir/scheduler.json"; do
        printf_with_spinner "Creating \"management_dir_$1/scheduler.json\" "
    done
    local ip="$(cat $management_dir/scheduler.json | grep -Eo tcp://\.*28786 | grep -o -P '(?<=tcp://).*(?=:28786)')"
    echo "${ip}"
}

#######################################
# Construct the Jupter server link based on the node IP adress and the jupyter link
#   Should work whether or not you have a token configured (only tested on token-having links)
#   Makes sure the link if set up with an IP, and not the node name (useful for e.g. quickly checking dask interface)
# Arguments:
#   1: The IP of the job
#   2: The jupyter link
#######################################
function clean_jupyter_link {
    local link_suffix="$(echo $2 | grep -oP '(?<=:11113/)'.* | head -1)"  # grep for anything after the port number
    if [ -z "$link_suffix" ] ;  # no suffix is found: no token, but also no 
    then
        printf "\nWarning: No token is set for this Jupyter server.\nVSCode will not be able to connect to this server, and notebooks running on this server can not be shared by link.\n">&2
    fi
    local jupyter_link="http://$1:11113/$link_suffix"
    echo $jupyter_link
}

args_precheck $#;  # check amount of arguments
job_name="$1"
jupyter_file="$MYBASEDIR/management_dir_$job_name/jupyter.txt"
ip="$(fetch_ip $job_name)";
link="$(fetch_jupyter_link $job_name)";
jupyter_link="$(clean_jupyter_link $ip $link)";

width="$(tput cols)"
printf '\r%*s' $width  # clear previous line
printf "\rJupyter server for \"$1\" is running at:
$jupyter_link
\n"
exit 0;
