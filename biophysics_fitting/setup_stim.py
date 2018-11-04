import neuron
from . import utils
h = neuron.h

def _append(cell, name, item):
    try:
        cell.name
    except AttributeError:
        cell.name = []
    cell.name.append(item)
    
def setup_soma_step(cell, amplitude = None, delay = None, duration = None):
    iclamp = h.IClamp(0.5, sec=cell.soma)
    iclamp.delay = delay # give the cell time to reach steady state
    iclamp.dur = duration # 5ms rectangular pulse
    iclamp.amp = amplitude # 1.9 ?? todo ampere
    _append(cell, 'iclamp', iclamp)

def setup_apical_epsp_injection(cell, dist = None, amplitude = None, delay = None):
    sec, x = utils.get_inner_section_at_distance(cell, dist)   
    iclamp2 = h.epsp(x, sec=sec)
    iclamp2.onset = delay
    iclamp2.imax = amplitude
    iclamp2.tau0 = 1.0 # rise time constant
    iclamp2.tau1 = 5 # decay time constant
    _append(cell, 'epsp', iclamp2)
    
def setup_bAP(cell):
    setup_soma_step(cell, amplitude = 1.9, delay = 295, duration = 5)
    
def setup_BAC(cell, dist = 970, delay = 295):
    setup_soma_step(cell, amplitude = 1.9, delay = delay, duration = 5) 
    setup_apical_epsp_injection(cell, dist = dist, amplitude = .5, delay = delay + 5)
   
def setup_StepOne(cell):
    setup_soma_step(cell, amplitude = 0.619, delay = 700, duration = 2000)

def setup_StepTwo(cell):
    setup_soma_step(cell, amplitude = 0.793, delay = 700, duration = 2000)

def setup_StepThree(cell):
    setup_soma_step(cell, amplitude = 1.507, delay = 700, duration = 2000)
