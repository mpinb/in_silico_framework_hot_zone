SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
shell=$SHELL
curl -fsSL https://pixi.sh/install.sh | bash
echo 'eval "$(pixi completion --shell {$SHELL})"' >> ~/.{$SHELL}rc
source ~/.bashrc
shell $SCRIPT_DIR/install_pandas_msgpack.sh
pixi install