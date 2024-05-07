# Installation
ISF requires `gcc` and `git` to be installed.

To install ISF on UNIX systems using the interactive installer, simply run:
```bash
./installer/install.sh
```

The installation will:
1. Download Anaconda for Python 2.7/3.8/3.9
2. Download and install conda dependencies
3. Download and install PyPI dependencies
4. Install `pandas-msgpack`
4. Patch `pandas-msgpack` (only for Py3.x) or `pandas` to support `CategoricalIndex` (only for Py2.7)
5. Install a corresponding ipykernel for notebooks
6. Compile all mechanisms in the [mechanisms](../mechanisms/) directory using `nrnivmodl`.

Tested on: 
| OS | `gcc` version | `git` version |
| -------- | -------- | -------- |
| `CentOS 7.9.2009`    | 4.8.5   | 2.31   |
| `Ubuntu 18.04.02 LTS`   | 7.5.0   | 2.42.0   |

## Additional installation configuration
It is possible to have an installation that includes nodejs for the Python 3.8 version. The interactive installer will prompt you.

## Non-interactive installation
To avoid interaction with the installer (for automated installation e.g.), you can also directly invoke the installer script that corresponds to the Python version you want to install:
```bash
./installer/py2.7/install.sh -p <installation_directory>
./installer/py3.8/install.sh -p <isntallation_directory> [--node]
./installer/py3.9/install.sh -p <installation_directory>
```