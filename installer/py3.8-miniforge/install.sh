#!/bin/bash
set -eE
set -o pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/utils.sh
# After this, the following variables/functions exist:
# Variables:
# - download_conda_flag
# - download_conda_packages_flag
# - download_pip_packages_flag
# - WOKRING_DIR
# - SCRIPT_DIR
# Functions:
# - print_title
# - usage
# - check_downloads
# - set_cli_args
# - check_prerequisites
# - setup_conda_in_shell
# - find_and_compile_mechanisms
# - patch_pandas_msgpack

# Global variables
echo "Script directory: ${SCRIPT_DIR}"
INSTALL_PATH=""
INSTALL_NODE=false
MINIFORGE="Miniforge3-$(uname)-$(uname -m).sh"
ENV_NAME="isf38"

# ---------- Read command line options ----------#
set_cli_args "$@"
INSTALL_PATH=$(realpath "$INSTALL_PATH")

# -------------------- 0. Setup -------------------- #
check_prerequisites
check_install_path $INSTALL_PATH
local_downloads_dir="$SCRIPT_DIR/downloads"
check_downloads $local_downloads_dir

# -------------------- 1. Installing Miniforge conda & mamba -------------------- #
print_title "1/5. Installing MiniForge conda & mamba"

# 1.0 -- Downloading miniforge (if necessary).
if [[ "${download_conda_flag}" == "true" ]]; then
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/$MINIFORGE"
fi

# 1.1 -- Installing miniforge
echo "Anaconda will be installed in: ${INSTALL_PATH}"
bash $MINIFORGE -b -p ${INSTALL_PATH}
# setup conda in current shell; avoid having to restart shell
setup_conda_in_shell ${INSTALL_PATH}

# init conda and mamba in current shell
source ${INSTALL_PATH}/etc/profile.d/conda.sh
source ${INSTALL_PATH}/etc/profile.d/mamba.sh 

echo "Creating environment $ENV_NAME"
# activate base env
source ${INSTALL_PATH}/bin/activate base


# -------------------- 2. Creating ISF environment -------------------- #
print_title "2/5. Creating ISF environment"
# create new env from base
mamba create -n $ENV_NAME -y python=3.8.5
mamba activate $ENV_NAME
mamba info
# install base packages
python -m pip --no-cache-dir install --no-deps -r $SCRIPT_DIR/pip_base_requirements.txt
# -- Installing nodejs if necessary
if [ "$INSTALL_NODE" = true ]; then
    echo "Installing nodejs"
    mamba install -y nodejs -c conda-forge --repodata-fn=repodata.json
fi

mamba env update -n $ENV_NAME -f $SCRIPT_DIR/environment.yml

# -------------------- 3. Patching pandas-msgpack -------------------- #
print_title "3/5. Installing & patching pandas-msgpack"
pip install cython==0.29.21  
patch_pandas_msgpack $SCRIPT_DIR

# -------------------- 4. installing the ipykernel -------------------- #
print_title "4/5. Installing the ipykernel"
python -m ipykernel install --name base --user --display-name $ENV_NAME

# -------------------- 5. Compiling NEURON mechanisms -------------------- #
print_title "5/5. Compiling NEURON mechanisms"
echo "Compiling NEURON mechanisms."
find_and_compile_mechanisms $SCRIPT_DIR $python_version_glob

