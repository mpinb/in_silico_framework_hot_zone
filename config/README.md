# Configuration

This directory contains example files on how you can configure your own experience with ISF.

## Necessary configurations
The `git` command needs to be configured in order to use ISF. We recommend `git v2.31`. Make sure you have `git` installed and your shell can find the command (you can check this by typing `which git`).

## Recommended configurations

As this codebase is for the most part a collection of modules, make sure ISF is added to your `PYTHONPATH`. That way, whatever Python you're using knows it should look in this folder for these modules. We recommend adapting your `~/.bashrc` file with the following lines (also defined in [bashrc.sh](./bashrc.sh)):
```shell
shopt -s expand_aliases
export MYBASEDIR="/gpfs/soma_fs/scratch/$USER"
module load git/2.31
ulimit -Sn "$(ulimit -Hn)"
export PYTHONPATH=$MYBASEDIR/project_src/in_silico_framework
export ISF_HOME=$MYBASEDIR/project_src/in_silico_framework
alias source_isf='source $MYBASEDIR/anaconda_isf2.7/bin/activate; export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH; cd $MYBASEDIR'
alias source_3='source $MYBASEDIR/anaconda_isf3.8/bin/activate; export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH; cd $MYBASEDIR'
alias source_33='source $MYBASEDIR/anaconda_isf3.9/bin/activate; export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH; cd $MYBASEDIR'
alias isf='cd $ISF_HOME'
alias wip='cd $MYBASEDIR/notebooks'
alias data='cd $MYBASEDIR'
alias cleanit='cd $MYBASEDIR; rm management_dir* -r; rm *slurm'
export PATH=$HOME/local/bin:$PATH
```

### Dask configuration
Dask automatically looks for files such as `distributed.yaml` throughout the entire codebase (like the one you find [here](./distributed.yaml)) for configuration. 

Dask automatically kill workers once they exceed 95% of the memory budget, which can be especially frustrating if you are doing a project where this is not a particular problematic thing to the point of terminating the worker without further ado (e.g. when working on a HPC). It can be useful to adapt this behavior. [This way](./distributed.yaml), you can configure workers to not be terminated or paused when they exceed memory 
```yml
distributed:
  worker:
    memory:
      target: 0.90  # target fraction to stay below
	  spill:  False # fraction at which we spill to disk
	  pause: False  # fraction at which we pause worker threads
	  terminate: False  # fraction at which we terminate the worker
```
(See this question on [StackOverflow](https://stackoverflow.com/questions/57997463/dask-warning-worker-exceeded-95-memory-budget))

### Port numbers
If you expect to be sharing the same machine/IP address (e.g. when sharing nodes on a HPC), you should ensure that you are not running any process on a port that is already in use. For this reason, it is useful to define your unique port numbers in [port_numbers.ini](./port_numbers.ini). Ports numbers can theoretically range from 0-99999, but some ports are used for default processes (e.g. `4040`, `8080`, `8000`...). Check which ports are in use by running any of the following commands on Linux systems:
```shell
lsof -i -P -n | grep LISTEN
netstat -tulpn | grep LISTEN
ss -tulpn | grep LISTEN
lsof -i:22 ## see a specific port such as 22 ##
nmap -sTU -O IP-address-Here
```
Or just trial and error your way to a port you like. The odds are generally in your favor.

## Automatic workflows
The workflow files in [.github/workflows](.github/workflows) define automatic workflows that run as soon as some trigger event happens. This trigger event is currently defined as push and pull requests on (almost) all branches. We have set up a local machine with (since last update) 8 runners and 10TB of disk space to take care of parallellized testing for speedy development. Upon such trigger event, it will perform the following actions:
1. Fetch the previous commit on the runner
2. Based on the commit it receives, figure out if a rebuild of the codebase is necessary
  a. Does a previous build already exists on the runner?
  b. Was the previous build succesful?
  c. Do the changes in the current commit not warrant a rebuild (i.e. they are not changes in the .github/workflows, testing/, or installer/ folder)
  If the answer is yes to all these, it will skip the build process, otherwise it will (re)build the codebase
3. Run the test suite

## Code coverage
The [codecov.yml](../.github/codecov.yml) file defines configuration for code coverage. It is currently set to allow all coverage differences when pushing to master. It can be setup to not allow merges with master if coverage drops, or does not improve by a certain amount etc.