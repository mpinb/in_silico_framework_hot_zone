import os
import re
from random import randrange
from definitions import ROOT_DIR
import transformation as tr


def get_am_paths_from_hx(hx_path, verbose=False):
    out = []
    with open(hx_path) as f:
        for l in f.readlines():
            if '${SCRIPTDIR}' in l:
                path = l.strip(' []').split(' ')[1]
                if verbose:
                    print path
                out.append(path)
    return out


def get_files_by_folder(path_to_folder, file_extension):
    return [path_to_folder + "/" + f for f in os.listdir(path_to_folder) if f.endswith(file_extension)]


def make_directories(path):
    if path is None:
        path = ROOT_DIR + "/output_" + str(randrange(100))
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_slice_name(am_path, image_file):
    am = os.path.basename(am_path)
    image_name = os.path.basename(image_file)
    spatial_graph_name = re.findall(r'[sS]\d+', am)[0]
    slice_name = int(re.findall(r'\d+', spatial_graph_name)[0])
    if spatial_graph_name in image_name:
        return slice_name
    else:
        return "No name found"


def get_file_name_from_path(path_to_file):
    return os.path.basename(path_to_file)


def get_am_image_match(am_paths, tif_paths):
    am_image_match = {}
    for am_path in am_paths:
        am_file_name = get_file_name_from_path(am_path)
        slice_name = re.findall(r'[sS]\d+', am_file_name)[0]
        for tif_path in tif_paths:
            tif_file_name = os.path.basename(tif_path)
            if slice_name in tif_file_name:
                am_image_match[am_path] = tif_path
    return am_image_match


def get_nearest_point(point, points):
    neighbours = get_neighbours_of_point(point, points)
    if neighbours is []:
        neighbours = points
    distances = [tr.get_distance(point, neighbour) for neighbour in neighbours]
    print distances
    nearest_point = neighbours[distances.index(min(distances))]
    return nearest_point


def get_neighbours_of_point(point, points, width=10):
    cube = [[axis - width, axis + width] for axis in point]
    # neighbours = [point for point in points for i in range(3) if cube[i][0] <= point[i] <= cube[i][1]]
    neighbours = [point for point in points if contains(point, cube)]
    return neighbours


def contains(point, cube):
    return [point[i] for i in range(3) if cube[i][0] <= point[i] <= cube[i][1]] == point

