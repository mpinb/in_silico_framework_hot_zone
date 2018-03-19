## Setup
 1. Download and install Anaconda 2.7 https://www.continuum.io/Downloads
 2. Install neuron such that you can import it as python module. 
     - Version 7.4 has been tested, newer versions are likely to work. 
     - Detailed installation instructions can be found here http://www.davison.webfactional.com/notes/installation-neuron-python/ or here https://www.neuron.yale.edu/phpBB/viewtopic.php?t=3489. 
     - If you use Ubuntu and you have trouble compling neuron, it has been reported that installing the following packages solves the problem: `sudo apt-get install bison flex g++ libxt-dev xorg-dev python-dev libncurses5-dev`
 3. Add the neuron folder to your PATH environment variable, such that you can run `nrnivmodl` anywhere
 3. Install the following dependencies:
    - sumatra, *used for parameterfiles*: `pip install sumatra`
    - pandas 0.19.2, *data analysis library*: `conda install pandas==0.19.2`
    - dask 0.16.1, *dynamic task scheduling and "big data" extension of pandas*: `conda install dask==0.16.1` #was 0.14.3
    - distributed 1.20.1 *allows non-bloccking computations and brings dask to a cluster*: `conda install distributed=1.20.1` #was 1.15.2
    - seaborn: *statistical data visualization*: `conda install seaborn==0.8.0`
    - fasterners: *robust file based locking*: `pip install fasteners`
    - jinja2: *html template engine, required for embedded animations*: `pip install jinja2`
 4. Clone or pull this repository: `git clone https://github.com/abast/in_silico_framework.git`. 
 5. Add your in_silico_framework folder to the PYTHONPATH variable
 6. Unzip the following folder: in_silico_framework/getting_started/barrel_cortex.zip such that the following file exists: `in_silico_framework/getting_started/barrel_cortex/nrCells.csv`
 6. Open the file model_data_base/simrun2/seed_manager and adjust the variable `path`. This will specify the location where used seeds are saved. Any location where you have write access is suitable.
 7. Run the test suite: `python run_tests.py`. 
 
Due to the statistical nature of the model, some tests might fail from time to time. These tests have the word _statistical_ in their description. If such a test fails, run the testsuite again. If that test fails again, there most likely is an issue. Tests, that do not have a _statistical_ flag in their description may never fail.

Run the following commands to install fast compression libraries:
- `conda install -c anaconda lz4`
- `conda install -c anaconda blosc`
- `conda install -c conda-forge python-blosc`
