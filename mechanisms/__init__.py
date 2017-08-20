import os, platform
import tables
import neuron
parent = os.path.abspath(os.path.dirname(__file__))

def compile_mechanisms():
    pass

arch = [platform.machine(), 'i686', 'x86_64', 'powerpc', 'umac']

try:
    assert(any([os.path.exists(os.path.join(parent, 'channels', a)) for a in arch]))
    assert(any([os.path.exists(os.path.join(parent, 'netcon', a)) for a in arch]))
except AssertionError:
    print "neuron mechanisms are not compiled." 
    print "Trying to compile them. Only works, if nrnivmodl is in PATH"
    os.system('(cd {path}; nrnivmodl)'.format(path = os.path.join(parent, 'channels')))
    os.system('(cd {path}; nrnivmodl)'.format(path = os.path.join(parent, 'netcon')))
    
    try:
        assert(any([os.path.exists(os.path.join(parent, 'channels', a)) for a in arch]))
        assert(any([os.path.exists(os.path.join(parent, 'netcon', a)) for a in arch]))
    except AssertionError:
        print "Could not complile mechanisms. Please do it manually"
        raise
    
    
neuron.load_mechanisms(os.path.join(parent, 'channels'))
neuron.load_mechanisms(os.path.join(parent, 'netcon'))