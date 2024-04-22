#!/bin/bash

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

SCRIPT_DIR=$(dirname "$0")
INVOCATION_DIR=$(pwd)

# Ask the user for the Python version
while true; do
    echo "Please enter the Python version (2.7, 3.8, or 3.9):"
    read PYTHON_VERSION

    # Validate the Python version
    if [[ "$PYTHON_VERSION" == "2.7" || "$PYTHON_VERSION" == "3.8" || "$PYTHON_VERSION" == "3.9" ]]; then
        break
    else
        echo "Invalid Python version. Please enter 2.7, 3.8, or 3.9."
    fi
done

# If Python version is 3.8, ask whether or not to install Node.js
if [ "$PYTHON_VERSION" == "3.8" ]; then
    while true; do
        echo "Do you want to install Node.js? (yes/no)"
        read INSTALL_NODEJS
        if [[ "$INSTALL_NODEJS" == "yes" || "$INSTALL_NODEJS" == "no" ]]; then
            break
        else
            echo "Invalid option. Please enter 'yes' or 'no'."
        fi
    done
fi

# Ask the user for the installation directory
while true; do
    echo "Please enter the installation directory:"
    read INSTALL_DIR

    # Validate the installation directory
    PARENT_DIR=$(dirname "$INSTALL_DIR")
    if [ -d "$PARENT_DIR" ]; then
        break
    else
        echo "Invalid directory. The parent directory ${PARENT_DIR} does not exist."
    fi
done

# Invoke the installation script in the correct folder
if [ "$INSTALL_NODEJS" == "yes" ]; then
    bash "${SCRIPT_DIR}/py${PYTHON_VERSION}/install.sh" -p "${INSTALL_DIR}" --node || exit 1;
else
    bash "${SCRIPT_DIR}/py${PYTHON_VERSION}/install.sh" -p "${INSTALL_DIR}" || exit 1;
fi