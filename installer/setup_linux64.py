#!/bin/python
''' This is an installer setting up a environment for the in_silico_framework
author: arco
date: 2018-08-31
'''

import sys
import os

basedir = sys.path[0]

print '*'*20
print 'running install script in %s' % basedir

print '*'*20
print '''usage: python install_all.sh --prefix=[where/it/should/be/installed] [--offline]

The path pointing to the NEURON installation is hardcoded in build_isf_from_anaconda2.py
todo: offer it as command line  option'''

print '*'*20
prefix = [a for a in sys.argv if a.startswith('--prefix=')]
if len(prefix) != 1:
    raise ValueError('You must provide a prefix, specifying, where you want to create the in_silico_framework environment')
else:
    prefix = prefix[0]
prefix = prefix[len('--prefix='):]
print 'prefix = %s' %prefix 

print '*'*20
print 'calling anaconda installer'
if not os.path.exists(os.path.join(basedir, 'Anaconda2-4.2.0-Linux-x86_64.sh'):
    os.system('cd %s; wget https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh')
if not os.system('cd %s; bash Anaconda2-4.2.0-Linux-x86_64.sh -b -p %s;' % (basedir, prefix)) == 0:
    raise RuntimeError

print '*'*20
print 'activate environment and install dependencies'
source_new_environment_path = os.path.join(prefix, 'bin', 'activate')
if not '--offline' in sys.argv:
	os.system('source %s; python build_isf_from_anaconda2_linux64.py;' % source_new_environment_path) 
else:
	os.system('source %s; python build_isf_from_anaconda2_linux64.py --offline;' % source_new_environment_path)
	
print '*'*20
print 'done! please check previous output for error messages. Please run the test_suite of in_silico_framework'
print 'To run the testsuite, cd into the in_silico_framework directory and run python run_tests.py'
print '*'*20
print 'you might want to modify your bashrc. I am using the following configuration:'
print "alias source_isf='source /abast/anaconda2_isf/bin/activate; export LD_LIBRARY_PATH=/abast/anaconda2_isf/lib:$LD_LIBRARAY_PATH; export PYTHONPATH=/nas1/Data_arco/project_src/in_silico_framework:$PYTHONPATH; export PATH=/nas1/Data_arco/prgr/nrn_isf_py2.7_florida/x86_64/bin:$PATH; cd /nas1/Data_arco'"
print '*'*20

