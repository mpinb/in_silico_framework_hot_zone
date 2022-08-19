#!/bin/bash
#
# This script can be used to install in-silico-framework (ISF) for Python 2 and 3
# Author: Omar Valerio (omar.valerio@mpinb.mpg.de)
# Date: 06.04.2022
# 
# 20.04.2022 - Modified the way pandas-msgpack is patch to support git 1.8.3.1 (soma)
#            - Clones in-silico-framework if the repository is missing
#            - Clones pandas-msgpack only if the repository is missing
#            - Download conda packages only if a new version exists (wget -N)
#            - Download Anaconda only if a new version exists (wget -N)              

usage() {
    cat << EOF
Usage: ./isf-install.sh [-d] -t {py2|py3} [-i <conda-install-path>] [-r <conda-requirements>] [-f <pip-requirements>]

    -h                          Display help
    -d                          Download in-silico-framework dependencies.

    -t py2|py3                  The target python version used by the installer.
    -i <conda-install-path>     The path where conda will be installed.
    -r <conda-requirements>     A file listing in-silico-framework conda packages.
    -f <pip-requirements>       A file listing in-silico-framework pip dependencies.
EOF
}

# Reading command line options supported by isf install script
while getopts "t:i:r:f:dh" opt; do
    case "${opt}" in
        t)
            target_python=${OPTARG}
            ;;
        i)
            conda_install_path=${OPTARG}
            ;;
        r)
            conda_requirements=${OPTARG}
            ;;
        f)
            pip_requirements=${OPTARG}
            ;;
        d)
            download_packages="true"
            ;;
        h)
            usage
            exit 0
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${target_python}" ] || ! [[ "${target_python}" =~ ^py2|py3$ ]] ; then
    echo "ERROR: The target python version is missing or an invalid value was provided."
    usage
    exit 1
else
    echo "In Silico Framework will be installed for ${target_python}"
fi


# STEP 0) Cloning In Silico Framework
echo "Cloning In Silico Framework"
ISF_HOME="$(pwd)/in_silico_framework"
if [ ! -r "${ISF_HOME}" ]; then
    #git clone https://github.com/research-center-caesar/in_silico_framework.git
    git clone https://github.com/abast/in_silico_framework.git
fi

# STEP 1) Downloading Anaconda
case "${target_python}" in
    "py2")
        echo "Downloading Anaconda2"
        wget https://repo.anaconda.com/archive/Anaconda2-4.2.0-Linux-x86_64.sh -N --quiet
        anaconda_installer="Anaconda2-4.2.0-Linux-x86_64.sh"
        #conda_requirements="${ISF_HOME}/etc/isf2-https-requirements.txt"
        #pip_requirements="${ISF_HOME}/etc/isf2-pip-requirements.txt"
        conda_requirements="isf2-https-requirements.txt"
        pip_requirements="isf2-pip-requirements.txt"
        conda_env_cmd="conda"
        ;;
    "py3")
        echo "Downloading Anaconda3"
        wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -N --quiet
        anaconda_installer="Anaconda3-2020.11-Linux-x86_64.sh"
        #conda_requirements="${ISF_HOME}/etc/isf3-https-requirements.txt"
        #pip_requirements="${ISF_HOME}/etc/isf3-pip-requirements.txt"
        conda_requirements="isf3-https-requirements.txt"
        pip_requirements="isf3-pip-requirements.txt"
        #conda_env_cmd="conda-env" #NOTE: doesn't work in offline mode
        conda_env_cmd="conda"
        ;;
esac

echo "[debug] anaconda_installer = ${anaconda_installer}"
echo "[debug] conda_requirements = ${conda_requirements}"
echo "[debug] pip_requirements = ${pip_requirements}"

# STEP 2) Installing Anaconda
echo "Installing Anaconda for ${target_python}"
if [ -z "${conda_install_path}" ]; then
    conda_install_path="${HOME}/conda-${target_python}"
fi
echo "Anaconda will be installed in: ${conda_install_path}"
bash ${anaconda_installer} -b -p ${conda_install_path}
source ${conda_install_path}/bin/activate

# STEP 3) Downloading In-Silico-Framework conda dependencies.
if [ ! -z "${conda_requirements}" ] && [ -r "${conda_requirements}" ]; then
    if [ ${download_packages} == "true" ]; then
        echo "Downloading conda packages using '${conda_requirements}'"
        [ -d conda_packages ] || mkdir conda_packages # conda packages download directory
        cat ${conda_requirements} | grep http | xargs -t -n 1 -P 8 wget -N -q -P conda_packages
        echo "Download conda packages completed."
    else
    	echo "Using existing/offline conda packages."
    fi
else
    echo "Error: The file ${conda_requirements} could not be read. Stopping."
    exit 1
fi

# STEP 4) Installing In-Silico-Framework conda dependencies.
echo "Installing In-Silico-Framework conda dependencies."
offline_requirements="isf-${target_python}-requirements.txt"
env_name="isf-${target_python}"
sed "s|https://.*/|$(pwd)/conda_packages/|" ${conda_requirements} > ${offline_requirements}
${conda_env_cmd} create --name ${env_name} --file ${offline_requirements} --quiet

# STEP 5) Activate python environment and install a recent version of pip
# NOTE: This version was pre-downloaded using:  python -m pip download pip
source activate ${env_name}
python -m pip install pip-20.3.4-py2.py3-none-any.whl

# STEP 6) Downloading In-Silico-Framework pip dependencies.
if [ ! -z "${pip_requirements}" ] && [ -r "${pip_requirements}" ]; then
    if [ ${download_packages} == "true" ]; then
        echo "Downloading pip dependencies using '${pip_requirements}'"
        [ -d pip_packages ] || mkdir pip_packages # pip packages download directory
        python -m pip download -r ${pip_requirements} -d pip_packages
        echo "Download pip packages completed."
    else
    	echo "Using existing/offline pip packages."
    fi
else 
    echo "Error: The file ${pip_requirements} could not be read. Stopping."
    exit 1
fi

# STEP 7) Installing In-Silico-Framework pip dependencies.
echo "Installing In-Silico-Framework pip dependencies."
python -m pip install -r ${pip_requirements} --no-index --find-links pip_packages

# STEP 8a) Patch pandas library (only for Python2)
if [ ${target_python} == "py2" ]; then
    echo "Patch pandas to support CategoricalIndex"
    python $ISF_HOME/installer/patch_pandas_linux64.py
fi

# STEP 8b) Patch dask library (only for Python3)
if [ ${target_python} == "py3" ]; then
    echo "Patch dask"
    python $ISF_HOME/installer/patch_dask_linux64.py
fi

# STEP 8c) Install modified pandas_msgpack (only Python3)
if [ ${target_python} == "py3" ]; then
    echo "Install modified pandas_msgpack"
    PD_MSGPACK_HOME="$(pwd)/pandas-msgpack"
    if [ ! -r "${PD_MSGPACK_HOME}" ]; then
        git clone https://github.com/abast/pandas-msgpack.git
    fi
    # Using Cython to generate and compile pandas-msgpack
    cd $PD_MSGPACK_HOME; python setup.py build_ext --inplace --force install
    pip list | grep pandas
fi

# STEP 9) Compiling neuron mechanisms.
echo "Compiling NEURON mechanisms."
cd $ISF_HOME/mechanisms/channels_${target_python}; nrnivmodl
cd $ISF_HOME/mechanisms/netcon_${target_python}; nrnivmodl
