Installation
============

ISF is available for Linux and macOS.

For installation and environment management, ISF uses [`pixi`](https://pixi.sh/latest/). 
You can install `pixi` by running:

.. code-block:: bash

   curl -fsSL https://pixi.sh/latest | bash

To install ISF with `pixi`, simply clone the repository and install with pixi:

.. code-block:: bash

   git clone https://github.com/mpinb/in_silico_framework.git --max-depth 1 &&
   cd in_silico_framework &&
   pixi install