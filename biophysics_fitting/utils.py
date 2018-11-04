def connected_to_dend_beyond(cell, sec, beyond_dist):
    if cell.distance_to_soma(sec, 1) > beyond_dist and sec.label in ('Dendrite', 'ApicalDendrite'):
        return True
    else:
        return bool(sum(connected_to_dend_beyond(cell, c, beyond_dist) 
                        for c in sec.children()
                        if sec.label in ('Dendrite', 'ApicalDendrite')))
        

def get_inner_sec_dist_list(cell):
    sec_dist_dict = {cell.distance_to_soma(sec, 1.0): sec 
                 for sec in cell.sections
                 if connected_to_dend_beyond(cell, sec, 1000)
                }
    return sec_dist_dict
    
def get_inner_section_at_distance(cell, dist):
    sec_dist_dict = get_inner_sec_dist_list(cell)
    dummy = {k - dist: v for k,v in sec_dist_dict.iteritems() if k > dist}
    closest_sec = dummy[min(dummy)]
    x = (dist - cell.distance_to_soma(closest_sec, 0.0)) / closest_sec.L
    return closest_sec, x
