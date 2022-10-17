# HOWTO SETUP IN-SILICO-FRAMEWORK

**Contents**

1. [Requirements](#requirements)
2. [Bash Profile Setup](#setup)
3. [Folder Structure](#structure)
4. [Clone In-Silico-Framework](#repository)
5. [Install In-Silico-Framework](#install)
6. [IPython Kernels](#kernels)
7. [Test Submit Jobs](#testing)
8. [Default Ports](#ports)
9. [Jupyter Extensions](#extensions)
10. [References](#references)

## Requirements

Every student needs to be able to synchronize their repository with https://github.com/abast/in_silico_framework


## Setup

Insert this lines into bash_profile, e.g. by using `vim ~/.bashrc`
Note: adapt the path mybasedir path for your cluster:

```bash
shopt -s expand_aliases
export MYBASEDIR="/gpfs/soma_fs/scratch/$USER"
module load git/2.31
ulimit -Sn "$(ulimit -Hn)"
export PYTHONPATH=$MYBASEDIR/project_src/in_silico_framework
export ISF_HOME=$MYBASEDIR/project_src/in_silico_framework
alias source_isf='source $MYBASEDIR/anaconda_isf2/bin/activate; export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH; cd $MYBASEDIR'
alias source_3='source $MYBASEDIR/anaconda_isf3/bin/activate; export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH; cd $MYBASEDIR'
alias isf='cd $ISF_HOME'
alias wip='cd $MYBASEDIR/notebooks'
alias data='cd $MYBASEDIR'
alias cleanit='cd $MYBASEDIR; rm management_dir* -r; rm *slurm'
export PATH=$HOME/local/bin:$PATH
```
Then run `source $HOME/.bashrc` to activate the new `.bashrc` file 

## Structure

We have a default folder structure, you need to create these folders:
- MYBASEDIR: typically on scratch
- MYBASEDIR/notebooks --> the contents should be kept in GitHub
- MYBASEDIR/results --> this is, where gnerated data goes
	--> for each notebook, there is (typically) one subfolder that is named exactly as the notebook is named
	--> notebooks start with the date they are created (YYYYMMDD)
- MYBASEDIR/prgr (optional), so far contains installers for anaconda, NEURON, ...

To replicate the folder structure run :

```bash
cd $MYBASEDIR
mkdir results
mkdir project_src
mkdir prgr
mkdir notebooks
```
## Repository

Clone `in_silico_framework` to `project_src`:

```bash
cd $MYBASEDIR/project_src
git clone https://github.com/abast/in_silico_framework
```

Note: you will be asked for the credentials of your user account. 
You will need to use your personal access token (PAT) instead of the password.

## Install

In order to install `in-silico-framework` for both Python 2 and Python 3 run :

```bash
cd $MYBASEDIR/project_src/in_silico_framework/installer
./isf-install.sh -d -t py2 -i $MYBASEDIR/anaconda_isf2
./isf-install.sh -d -t py3 -i $MYBASEDIR/anaconda_isf3
```

## Kernels

After the installation is completed, proceed to install the IPython Kernels:

```bash
source_isf; python -m ipykernel install --name root --user --display-name isf2
source_3; python -m ipykernel install --name base --user --display-name isf3
```

## Testing

Check if you can submit a job:

```bash
source_isf
sbatch project_src/in_silico_framework/SLURM_scripts/launch_dask_on_SOMA_gpu_interactive_1.sh  session1
```
**Note:** Currently the launching scripts work only with `in-silico-framework` (Python 2)

Make sure that the job appears with `Running` (R) status in slurm job queue:

```bash
squeue -u $USER
```

Look at the corresponding management dir to find the ip address where `jupyter` is running:

```bash
cat management_dir_session1/scheduler.json
```
Use putty to open a SOCKS-5 proxy to the login node (somalogin01 or somalogin02)
 - you can do this with the -D option of the `ssh` command
    - example: `ssh -D 4040 somalogin02`
 - Open firefox settings, search for proxy and activate SOCKS-5 proxy. IP: 127.0.0.1, PORT: 4040 (or the port number passed to the ssh command)
 - this is explained in more detail in chantals google doc: https://docs.google.com/document/d/1S0IM7HgRsRdGXN_WFeDqPMOL3iDt1Obosuikzzc8YNc/edit#heading=h.dbift17bl1rt


## Ports

We have default ports
- 11112: jupyter notebook
- 11113: jupyter lab
- 28787: python 2 scheduler interface
- 28786: python 2 scheduler
- 38787: python 3 scheduler interface
- 38786: python 3 scheduler

## Extensions

We have a set of default extensions to `jupyter notebook`

- Codefolding in Editor
- Collapsible Headings
- contrib_nbextensions_help_item
- Nbextensions_dashboard tab
- Notify
- Codefolding
- Initialization cells
- Nbextensions edit menu item
- Table of Contents (2)

To install the extensions for `jupyter notebook` enter:

```bash
source_isf
jupyter contrib nbextension install --user
jupyter nbextensions_configurator enable --user
```

## References

--> located at Z:\Share\HowTos\How-to use the AXON cluster.txt
--> https://docs.google.com/document/d/1S0IM7HgRsRdGXN_WFeDqPMOL3iDt1Obosuikzzc8YNc/edit#heading=h.dbift17bl1rt