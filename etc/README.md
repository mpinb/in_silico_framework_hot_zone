# ETC

Tis directory provides the following various subdirectories:
- amira_utils: everything you need to work with AMIRA
- cluster_control: for when you have a server that does not provide SLURM job management
- SLURM_scripts: handy scripts that automate many SLURM jobs, such as setting up DASK, requesting resources...
- Figures: Just some figures to spice up the main page

Apart from these folders, you may also find Python environment specifications. These are simply a list of conda and pip packages for both he Python3 and Python2.7 environments for ISF. These are used in the github workflows at .github/workflows to install the correct dependencies to run tests.

These are not in .github/, since conda is being real annoying about from where and how it can read module lists. Moving these files to .github/ is a good way to break the environment for some reason. Probably due to the fact that .github starts with a leading dot.