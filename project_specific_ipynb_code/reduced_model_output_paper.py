def get_surround_whiskers(whisker):
    '''arguments:
        whisker: str (in the format "A1", "C4", "Delta")
    returns:
        a list containing all surround whiskers for the provided whisker.'''
    layout = [['A1', 'A2', 'A3', 'A4'], ['B1', 'B2', 'B3', 'B4'], ['C1', 'C2', 'C3', 'C4'], ['D1', 'D2', 'D3', 'D4'], ['E1', 'E2', 'E3', 'E4']]
    greeks = ['Alpha', 'Beta', 'Gamma', 'Delta']
    greeks_lookup = dict(zip(greeks, [['A1', 'B1', 'Beta'], ['Alpha', 'Gamma', 'B1', 'C2'], ['Beta', 'Delta', 'C1', 'D1'], ['Gamma', 'D1', 'E1']]))
    if whisker in greeks:
        sws = greeks_lookup[whisker]
        return sws
    
    row_index = [layout.index(l) for l in layout if whisker in l][0]
    row_range = [row_index - 1 if row_index - 1 >= 0 else 0, row_index + 1 if row_index + 1 <= 4 else 4]
    
    arc_index = int(whisker[1])-1
    arc_range = [arc_index - 1 if arc_index - 1 >= 0 else 0, arc_index + 1 if arc_index + 1 <= 3 else 3]
    
    sws = []
    for row in range(row_range[0], row_range[1]+1):
        for arc in range(arc_range[0], arc_range[1]+1):
            sws.append(layout[row][arc])
    if arc_index == 0: # need to add greeks to surround
        sws.extend(greeks[row_index-1 if row_index-1 >= 0 else 0:row_index+1])
    
    sws.remove(whisker)
    return sws