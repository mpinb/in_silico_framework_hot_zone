<div align="center">

<img src=./docs/_static/_images/isf-logo-white.png#gh-dark-mode-only width='350'>
<img src=./docs/_static/_images/isf-logo-black.png#gh-light-mode-only width='350'>

# The In Silico Framework (ISF)
[![Linux](https://github.com/mpinb/in_silico_framework/actions/workflows/test-isf-py38-pixi-linux.yml/badge.svg)](https://github.com/mpinb/in_silico_framework/actions/workflows/test-isf-py38-pixi-linux.yml)
[![OSX](https://github.com/mpinb/in_silico_framework/actions/workflows/test-isf-py38-pixi-macos.yml/badge.svg)](https://github.com/mpinb/in_silico_framework/actions/workflows/test-isf-py38-pixi-macos.yml)
[![codecov](https://codecov.io/gh/mpinb/in_silico_framework/graph/badge.svg?token=V4P4QMFM12)](https://codecov.io/gh/mpinb/in_silico_framework)

</div>

An In Silico Framework for multi-scale modeling and analysis of *in vivo* neuron-network mechanisms

## Documentation

Web-hosted documentation is available [here](https://wwwuser.gwdguser.de/~b.meulemeester/index.html)

You can self-host using

```bash
pixi r build_docs
pixi r host_docs
```

## Installation

This version of ISF is available for Linux and macOS.

For installation and environment management, ISF uses [pixi](pixi.sh). You can install pixi by running:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

You can then install this version of ISF by running:
```bash
git clone https://github.com/mpinb/in_silico_framework_hot_zone.git --depth 1 &&
cd in_silico_framework_hot_zone &&
pixi install
```

## Usage

Launch a jupyter server:
```bash
pixi r launch_jupyter_server
```

Launch a dask server and workers for parallel computing (optional, but recommended):
```bash
pixi r launch_dask_server
```
```bash
pixi r launch_dask_workers
```

You can then connect to the jupyter server in your browser and start coding using ISF.

## Tutorials

Please visit the tutorials (either in `getting_started/tutorials` or on the documentation page) for a walkthrough of ISF's most important workflows.
