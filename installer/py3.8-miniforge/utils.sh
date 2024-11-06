set -eE
set -o pipefail

function check_prerequisites {
    # Check if git is available
    if ! command -v git &> /dev/null; then
        echo "git could not be found. Please install git."
        exit 1
    fi

    # Check if gcc is available
    if ! command -v gcc &> /dev/null; then
        echo "gcc could not be found. Please install gcc."
        exit 1
    fi
}

pushd . # save this dir on stack

# Global variables
WORKING_DIR=$(pwd)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "Script directory: ${SCRIPT_DIR}"

function print_title {
    local str="$1"
    local pad_char="-"
    local width="$(tput cols)"
    local str_length=${#str}
    local padding=$(( (width - str_length - 2) / 2 ))  # Calculate padding for centering

    # Print top border
    printf '%*s\n' "$width" | tr ' ' "$pad_char"

    # Print title with padding
    printf '%*s' "$padding" | tr ' ' " "
    printf ' %s ' "$str"
    printf '%*s\n' "$padding" | tr ' ' " "

    # Print bottom border
    printf '%*s\n' "$width" | tr ' ' "$pad_char"
}

function usage {
    cat << EOF
Usage: install [-p|--path <conda-install-path>] [--node]

    -h | --help     Display help
    -p | --path     The path where the conda environment conda will be installed.
    --node 			Install nodejs along with the python environment
EOF
}

function check_downloads {
    local_downloads_dir=$1
    
    # Check if the downloads directory exists
    if [ ! -d "$local_downloads_dir" ]; then
        echo "No downloads directory found. Created one to download packages in."
        mkdir $local_downloads_dir
    fi

    # 0.1 -- Create downloads folder (if it does not exist)
    if [ ! -d "$local_downloads_dir" ]; then
        mkdir $local_downloads_dir
    fi

    # 0.2 -- Check 2: Are the conda packages already downloaded?
    if [ ! -d "$local_downloads_dir/conda_packages" ]; then
        echo "No conda_packages directory found in downloads. Created one to download conda packages in."
        mkdir $local_downloads_dir/conda_packages # conda packages download directory
        download_conda_packages_flag="true"
    elif [ ! "$(ls -A $local_downloads_dir/conda_packages)" ]; then
        echo "No conda packages found in $local_downloads_dir/conda_packages. They will be downloaded."
        download_conda_packages_flag="true"
    else
        echo "Warning: found pre-existing packages in $local_downloads_dir. They will not be redownloaded. If you have changed the conda_requirements.txt file or pip_requirements.txt, you should remove this folder or its contents before attemtping a reinstall."
        download_conda_packages_flag="false"
    fi

    # 0.3 -- Check 3: Are the pip packages already downloaded?
    if [ ! -d "$local_downloads_dir/pip_packages" ]; then
        echo "No pip_packages directory found in downloads. Created one to download PyPI packages in."
        mkdir $local_downloads_dir/pip_packages  # pip packages download directory
        download_pip_packages_flag="true"
    elif [ ! "$(ls -A $local_downloads_dir/pip_packages)" ]; then
        echo "No PyPI packages found in downloads/pip_packages. They will be downloaded."
        download_pip_packages_flag="true"
    else
        echo "Warning: found PyPI packages in downloads/pip_packages. They will not be redownloaded. If you have changed the pip_requirements.txt file, you should remove this folder or its contents before attemtping a reinstall."
        download_pip_packages_flag="false"
    fi

    # 0.4 -- Check 4: Are there any downloads necessary?
    if [[ "${download_conda_flag}" == "false" && "${download_conda_packages_flag}" == "false" && "${download_pip_packages_flag}" == "false" ]]; then
        echo "No downloads necessary."
    fi
}

function set_cli_args {
    while [ "${1:-}" != "" ]; do
    case "$1" in
        "-p" | "--path")
        shift
        INSTALL_PATH="$1"
        ;;
        "--node")
        INSTALL_NODE=true
        ;;
        "-h" | "--help")
        usage
        exit 0
        ;;
    esac
    shift
    done
    shift $((OPTIND-1))
    if [ -z $INSTALL_PATH  ]; then
        echo 'Missing -p or --path. Please provide an installation path for the environment.' >&2
        exit 1
    fi
}

function check_install_path {
    install_path=$1
    parent_dir=$(dirname "$install_path")
    if [ ! -d "$parent_dir" ]; then
        echo "Error: Parent directory $parent_dir does not exist." >&2
        exit 1
    fi
}

function setup_conda_in_shell {
    install_path=$1
    __conda_setup="$($install_path/bin/conda 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        # if bash hook was successful, evaluate it
        eval "$__conda_setup"
    else
        # bash hook unsuccessful
        if [ -f "$install_path/etc/profile.d/conda.sh" ]; then
            # simply source the conda script
            . "$install_path/etc/profile.d/conda.sh"
        else
            # export PATH="$install_path/bin:$PATH"
            echo "ERROR: conda bash hook unsuccessful and conda.sh not found. Was the installation succesful?"
            echo "Exiting."
            exit 1
        fi
    fi
    unset __conda_setup

    # For miniforge: also setup mamba
    if [ -f "$install_path/etc/profile.d/mamba.sh" ]; then
        . "$install_path/etc/profile.d/mamba.sh"
    fi
}

function verify_neuron_compilation {
    compilation_dir=$1
    if [ -d "$compilation_dir" ]; then
        SO_FILE=$(find "$compilation_dir" -name "*.so" -print -quit)
        if [ ! -f "$SO_FILE" ]; then
            echo "$compilation_dir does not contain a .so file. Compilation was unsuccesful. Please inspect the output of nrnivmodl for further information."
            exist 1;
        fi 
    else
        echo "No directory found containing *.c files. Compilation was unsuccesful."
        exit 1;
    fi
}

function find_and_compile_mechanisms {
    base_dir=$1
    python_version_glob=$2
    shopt -s extglob
    for d in $(find $base_dir/../../mechanisms/*/*$python_version_glob* -maxdepth 1 -type d)
    do
        # find .mod files
        if [ $(find $d -maxdepth 1 -name "*.mod" -print -quit) ]; then
            echo "compiling mechanisms in $d"
            cd $d;
            
            # Find directory containinc .c files
            COMPILATION_DIR=$(find $d -mindepth 2 -type f -name "*.c" -printf '%h\n' | head -n 1 || true)
            if [ -d "$COMPILATION_DIR" ]; then
                echo "Found previously created compilation directory ${COMPILATION_DIR}"
                echo "Deleting previously created $COMPILATION_DIR "
                rm -r $COMPILATION_DIR
            fi
            
            # compile in the directory
            output=$(nrnivmodl 2>&1)
            if echo "$output" | grep -iq "error"; then
                echo "$output"
                exit 1
            else
                echo "$output"
            fi

            # Verify if compilation was succesful
            COMPILATION_DIR=$(find $d -type f -name "*.c" -printf '%h\n' | head -n 1 || true)
            verify_neuron_compilation $COMPILATION_DIR
        fi
    done
}

function patch_pandas_msgpack {
    script_dir=$1
    PD_MSGPACK_HOME="$script_dir/pandas-msgpack"
    pushd .
    if [ ! -d "${PD_MSGPACK_HOME}" ]; then
        cd $script_dir
        echo "Cloning pandas-msgpack from GitHub."
        git clone https://github.com/abast/pandas-msgpack.git
        git -C $script_dir/pandas-msgpack apply $script_dir/pandas_msgpack.patch
    fi
    cd $PD_MSGPACK_HOME; python setup.py build_ext --inplace --force install
    pip list | grep pandas
    popd
}