# please activate your enviroment before running this script!

# change this path such that it points to a nrnpython installation
import os
import sys
NEURON_setup_py_dir = '/nas1/Data_arco/prgr/nrn_isf_py2.7_florida/src/nrnpython'
basedir = os.path.dirname(__file__)#sys.argv[0]

if '--offline' in sys.argv:
        # conda packages
	pkgs_dir_conda = os.path.join(basedir, 'offline_pkgs', 'conda')
	names = sorted([f for f in os.listdir(pkgs_dir_conda) if f.endswith('.bz2')])
	commands_conda = ['cd {}; conda install -y --offline {}'.format(pkgs_dir_conda, n) for n in names]
        
        # pip packages
        pkgs_dir_pip = os.path.join(basedir, 'offline_pkgs', 'pip')
        names = sorted([f for f in os.listdir(pkgs_dir_pip) if (f.endswith('.gz') or f.endswith('whl'))]) 
        commands_pip = ['cd {}; pip install {} --no-deps'.format(pkgs_dir_pip, n) for n in names]

        commands = commands_conda + commands_pip
else:
	commands = [
		#'conda install -y -c anaconda libstdcxx-ng=7.2.0=h7a57d05_2', # maybe not necessary #
		#python'conda install -y -c anaconda gcc', #maybe not necessary
		'conda install -y -c anaconda libgcc-ng=7.2.0=h7cc24e2_2', # maybe not necessary #
		'conda install -y -c free git=2.11.1=0',
		'conda install -y -c free snappy=1.1.6=0', 
		'conda install -y -c free zlib=1.2.11=0',
		'conda install -y -c free lz4=0.10.1=py27_0',
		#'conda install -y -c anaconda blosc=1.12.0=he42ba99_0',
		'conda install -y -c free c-blosc=1.10.2',
		#'conda install -y -c anaconda python-blosc=1.4.4=py27_0',	
		'conda install -y -c free python-blosc=1.5.1=py27_0',
		'conda install -y -c free scikit-learn=0.18.1=np111py27_1',            
		'conda install -y -c free pandas==0.19.2=np111py27_1',
		'conda install -y -c free libpng=1.6.30=1',		
		'conda install -y -c free matplotlib=2.0.2=np111py27_0',
		#'conda install -y -c free graphviz=2.38.0=2',	
		'conda install -y -c free python-graphviz=0.5.2=py27_1',
		'conda install -y -c free libgcc=5.2.0=0', # put at the end to make sure we have a florida compatible libstdc++.so.6.0 --> libstdc++.so.6.0.21
		'conda install -y -c conda-forge jupyter_contrib_nbextensions', # this does something within jupyte-notebook and needs libstdc++.so.6 --> put it after libgcc		
		'conda install -y -c free cloudpickle=0.2.2=py27_0',
        'conda install -y -c free tornado=4.5.1=py27_0',
		'conda install -y -c free msgpack-python=0.4.8=py27_0',
		'conda install -y -c free distributed=1.22.1=py27_0',
		'conda install -y -c free dask=0.18.2=py27_0',
		'conda install -y -c free seaborn==0.8.0',
		'conda install -y -c free mock=2.0.0=py27_0',
		'pip install fasteners==0.14.1 sumatra==0.7.4',
		'pip install bluepyopt==1.6.4 deap==1.2.2'
		]

#conda install jupyter_client=5.0.1
#conda install pyzmq=16.0.2=py27_0

for c in commands:
    print('*'*20)
    print(c)
    os.system(c)

# patch pandas
# this adds support for CategoricalIndex for msgpack format
# by arco, 208-08-30

print('patching pandas to support CategoricalIndex in msgpack format')
import os
import pandas.io.packers
dest_path = os.path.dirname(pandas.io.packers.__file__)
dest_path = os.path.join(dest_path, 'packers.py')
source_path = os.path.join(basedir, 'pandas_patch', 'packers.py')
with open(dest_path, 'w') as f_dest:
    with open(source_path) as f_source:
		f_dest.write(f_source.read())
		
print('you need a installation of NEURON. I try to install from the specified directory ...')
print('installing NEURON')
print('In case, neuron does not work, please go to your neuron directory and recompile it, ideally, while THIS ENVIRONMENT IS ACTIVATED')

# wget 
# ./configure --prefix=/nas1/Data_arco/prgr/nrn_isf_py2.7_florida --with-nrnpython --without-iv
# make 
# make install

os.system('cd {}; python setup.py install'.format(NEURON_setup_py_dir))
