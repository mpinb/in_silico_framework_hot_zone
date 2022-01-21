# IMPORTANT: Activate conda environment before running the script.
import os
basedir = os.getcwd()


def system(x):
    os.system('/bin/bash -c "' + x + '"')


# pandas_msgpack is currently the only package installed using setup.py
names = [f for f in os.listdir(basedir) if os.path.isdir(f) and 'setup.py' in os.listdir(os.path.join(basedir, f))]
commands = ['cd {}; python setup.py install'.format(os.path.join(basedir, n)) for n in names]

for c in commands:
    print('[setup] {}'.format(c))
    system(c)

