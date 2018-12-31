import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if (nb_dir not in sys.path):
    sys.path.append(nb_dir)

import transformTools as tr

amDataPath = str('../data/neuron1/hoc/')
outputFolderPath = str('../output/neuron1/hoc/')

sampleFile = amDataPath + "500_GP_WR639_cell_1547_SP5C_checked_RE.hoc"

set1 = tr.read.hocFile(sampleFile)
numberOfMatches = 4

points = tr.matchPoints(set1, set1, numberOfMatches) 
distances = tr.getDistance.nodes(points)
uniqEdge1 = max(distances)
distances.remove(uniqEdge1)
distances.remove(uniqEdge1)
uniqEdge2 = max(distances)

print(uniqEdge1, uniqEdge2)

