'''
Created on Nov 01, 2018

@author: abast
'''

import numpy as np
from functools import partial

####################################
# selection of sections
####################################
def connected_to_structure_beyond(cell, sec, beyond_dist, struct_list = ['Dendrite', 'ApicalDendrite']):
    if cell.distance_to_soma(sec, 1) > beyond_dist and sec.label in struct_list:
        return True
    else:
        return bool(sum(connected_to_dend_beyond(cell, c, beyond_dist) 
                        for c in sec.children()
                        if sec.label in struct_list))
        
connected_to_dend_beyond = partial(connected_to_structure_beyond, struct_list = ['Dendrite', 'ApicalDendrite'])

def get_inner_sec_dist_list(cell, beyond_dist = 1000, beyond_struct = ['ApicalDendrite']):
    '''returns sections, that are connected to compartments with labels in beyond_struct that have a minimum soma distance of
    beyond_dist. This is useful to get sections of the apical trunk filtering out oblique dendrites.'''
    sec_dist_dict = {cell.distance_to_soma(sec, 1.0): sec 
                 for sec in cell.sections
                 if connected_to_structure_beyond(cell, sec, beyond_dist, ['ApicalDendrite'])
                }
    return sec_dist_dict
    
def get_inner_section_at_distance(cell, dist, beyond_dist = 1000, beyond_struct = ['ApicalDendrite']):
    '''Returns the section and relative position of that section, such that the soma distance (along the dendrite) is dist.
    Also, it is assured, that the section returned has children that have a soma distance beyond beyond_dist of the label in
    beyond_struct'''
    sec_dist_dict = get_inner_sec_dist_list(cell, beyond_dist, beyond_struct)
    dummy = {k - dist: v for k,v in sec_dist_dict.iteritems() if k > dist}
    closest_sec = dummy[min(dummy)]
    x = (dist - cell.distance_to_soma(closest_sec, 0.0)) / closest_sec.L
    return closest_sec, x

#####################################
# read out Vm at section
#######################################

def tVec(cell):
    return np.array(cell.tVec)

def vmSoma(cell):
    return np.array(cell.soma.recVList[0])

def _get_apical_sec_and_i_at_distance(cell, dist):
    sec, target_x = get_inner_section_at_distance(cell, dist)
    # roberts code to get closest segment
    mindx  = 1
    for i in range(len(sec.segx)):
        dx = np.abs(sec.segx[i]-target_x)
        if dx < mindx:
            mindx = dx
            minSeg = i
    return sec, mindx, minSeg

def vmApical(cell, dist = None):
    assert(dist is not None)
    sec, mindx, minSeg = _get_apical_sec_and_i_at_distance(cell, dist)
    return np.array(sec.recVList[minSeg])  

def vmApical_position(cell, dist = None):
    sec, mindx, i = _get_apical_sec_and_i_at_distance(cell, dist)
    target_x = [seg for seg in sec][i].x
    print target_x
    
    minDist = 1e9
    closest_point_to_section = None
    for j in range(len(sec.pts)):
        pt = sec.pts[j]
        ptx = sec.relPts[j]
        dx = abs(ptx-target_x)
        if dx < mindx:
            closest_point_to_section = pt
            mindx = dx     
    return closest_point_to_section

