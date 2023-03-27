# Documentation

Documentation is built automatically using Sphinx. The [configuration file](./conf.py) sets up all the requirements for generating documentation on the fly. Most notably, it enables autosummary, which scans the codebase for docstrings and uses this for the HTML.

Generating documentation on the soma cluster can be done by using Sphinx in an Apptainer. Start a shell in the sphinx apptainer and run doc generating commands like so:
```console
pip install furo
apptainer shell -H /gpfs/soma_fs/home/meulemeester/ -B/gpfs/soma_fs/scratch/meulemeester:/gpfs/soma_fs/scratch/meulemeester /gpfs/soma_fs/soft/CentOS_7/packages/x86_64/containers/sphinx/sphinx_5.3.0.sif
source ~/.bashrc
cd project_src/in_silico_framework/docs
make clean
make html
```