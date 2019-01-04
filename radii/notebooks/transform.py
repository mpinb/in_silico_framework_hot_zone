import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if (nb_dir not in sys.path):
    sys.path.append(nb_dir)

import transformTools as tr
import radii as radi

hocDataPath = str('../data/neuron1/hoc/')
amDataPath = str('../data/neuron1/am_final/')
outputFolderPath = str('../output/neuron1/hoc/')

hocFile = hocDataPath + "500_GP_WR639_cell_1547_SP5C_checked_RE.hoc"
amFile = amDataPath + 'final_spatial_graph_with_radius_data.am'

set1 = tr.read.hocFile(hocFile)

hocFile = '/home/amir/Projects/radii/radii/data/neuron1/landmark/hocFile.txt'
with open(hocFile, 'w') as f:
    for item in set1:
        f.write('{:f}\t{:f}\t{:f} \n'.format(item[0], item[1], item[2]))

# points = radi.spacialGraph.getSpatialGraphPoints(amFile)
# set2 = list(map(lambda x: map(lambda y: int(y/0.092), x), points))

numberOfEdges = 2

# matchedSet = tr.getDistance.matchEdges(set1, set2, numberOfEdges)

# print(matchedSet)
