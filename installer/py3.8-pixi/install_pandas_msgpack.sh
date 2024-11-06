#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PD_MSGPACK_HOME="$SCRIPT_DIR/pandas-msgpack"
pushd .
if [ ! -d "${PD_MSGPACK_HOME}" ]; then
    cd $SCRIPT_DIR
    echo "Cloning pandas-msgpack from GitHub."
    git clone https://github.com/abast/pandas-msgpack.git
    git -C $SCRIPT_DIR/pandas-msgpack apply $SCRIPT_DIR/pandas_msgpack.patch
fi
cd $PD_MSGPACK_HOME; python setup.py build_ext --inplace --force install
pip list | grep pandas
popd