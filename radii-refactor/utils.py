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
    nearest_point = neighbours[distances.index(min(distances))]
    return nearest_point


def get_neighbours_of_point(point, points, width=10):
    cube = [[axis - width, axis + width] for axis in point]
    # neighbours = [point for point in points for i in range(3) if cube[i][0] <= point[i] <= cube[i][1]]
    neighbours = [point for point in points if contains(point, cube)]
    return neighbours


def contains(point, cube):
    if [point[i] for i in range(3) if cube[i][0] <= point[i] <= cube[i][1]] == point:
        return True
    else:
        return False


def copyAmFilesToOutputFromHxPath(hx_path, relpath_list, output):
    make_directories(output)
    hx_folder = os.path.dirname(hx_path)
    for path in relpath_list:
        path = path.replace('${SCRIPTDIR}/', '')
        reldir = os.path.dirname(path)
        if len(reldir) > 0:
            make_directories(os.path.join(output, reldir))
        I.shutil.copy(os.path.join(hx_folder, path), I.os.path.join(output, path))
    I.shutil.copy(hx_path, output)


def copyMaxZFilesToOutput(maxZ_folders, output):
    '''assumes, that max z projections are in folders, which cointain a string like S001, indicating the corresponding slice.
       Further assumes, that the file is precisely named "max_z_projection.tif.
       Copies max z projections to output
       Renames them to foldername + '_max_z_projection.tif' '''
    make_directories(output)
    for zFolder in maxZ_folders:
        zfolderName = I.os.path.basename(zFolder)

        I.shutil.copy(I.os.path.join(zFolder, "max_z_projection.tif"), output)

        dst_file = os.path.join(output, "max_z_projection.tif")
        new_file_name = zfolderName + "_max_z_projection.tif"
        new_file_name = new_file_name.replace("0", "", 1)
        new_dst_file_name = os.path.join(output, new_file_name)
        os.rename(dst_file, new_dst_file_name)


def createTclFileForConvertingToAmiraAscii(amFiles, tclFile, port=7175, host='localhost'):
    # host = 'localhost'
    # port = 7175

    f = open(tclFile, "w")
    f.write("set server {0}\n".format(host))
    f.write("set sockChan [socket $server {0}]\n".format(port))

    for amFile in amFiles:
        loadedFile = os.path.basename(amFile)
        f.write("puts $sockChan \"load {0}\"\n".format(amFile))

        C_puts_sockChan = 'puts $sockChan'
        C_exportData = " exportData "
        Q_AmiraMesh_ascii_format = '"AmiraMesh ascii SpatialGraph" '
        Q_loadedFile = '"' + loadedFile + '"'
        Q_amFilePath = '"' + amFile + '"'

        f.write(C_puts_sockChan + " {" + Q_loadedFile + C_exportData + Q_AmiraMesh_ascii_format + Q_amFilePath + " }\n")
        f.write("puts $sockChan \"remove {}\"\n".format(loadedFile))


def createTclFileForMergingAmFiles(amFiles, mergeAmFile, tclFile, port=7175, host='localhost'):
    # host = 'localhost'
    # port = 7175

    C_puts_sockChan = 'puts $sockChan'
    C_merge = " merge "
    C_exportData = " exportData "
    Q_AmiraMesh_ascii_format = '"AmiraMesh ascii SpatialGraph" '
    C_applyTransform = " applyTransform "

    seedAmFilePath = amFiles[0]
    seedAmFile = os.path.basename(amFiles[0])
    Q_mergeAmFile = '"' + mergeAmFile + '"'
    Q_seedAmFile = '"' + seedAmFile + '"'

    f = open(tclFile, "w")
    f.write("set server {0}\n".format(host))
    f.write("set sockChan [socket $server {0}]\n".format(port))
    f.write("puts $sockChan \"load {0}\"\n".format(seedAmFilePath))  ## comment for hx file
    f.write(C_puts_sockChan + " {" + Q_seedAmFile + C_applyTransform + "}\n")

    for amFile in amFiles[1:]:
        loadedFile = os.path.basename(amFile)
        f.write("puts $sockChan \"load {0}\"\n".format(amFile))  ## comment for hx file

        Q_LoadedFile = '"' + loadedFile + '"'

        f.write(C_puts_sockChan + " {" + Q_seedAmFile + C_merge + Q_LoadedFile + "}\n")
        f.write(C_puts_sockChan + " {" + Q_LoadedFile + C_applyTransform + "}\n")
        f.write("puts $sockChan \"remove {}\"\n".format(loadedFile))

    f.write(C_puts_sockChan + " {" + Q_seedAmFile + C_exportData + Q_AmiraMesh_ascii_format + Q_mergeAmFile + " }\n")
