import Interface as I
import os


def get_am_paths_from_hx(hx_path, verbose=False):
    out = []
    with open(hx_path) as f:
        for l in f.readlines():
            if '${SCRIPTDIR}' in l:
                path = l.strip(' []').split(' ')[1]
                # abspath = relpath.replace('${SCRIPTDIR}', I.os.path.dirname(hx_path))
                if verbose:
                    print path
                out.append(path)
    return out
    # print f.read()[:10000]


def my_makedirs(path):
    if not I.os.path.exists(path):
        I.os.makedirs(path)


def copyAmFilesToOutputFromHxPath(hx_path, relpath_list, output):
    my_makedirs(output)
    hx_folder = I.os.path.dirname(hx_path)
    for path in relpath_list:
        path = path.replace('${SCRIPTDIR}/', '')
        reldir = I.os.path.dirname(path)
        if len(reldir) > 0:
            my_makedirs(I.os.path.join(output, reldir))
        I.shutil.copy(I.os.path.join(hx_folder, path), I.os.path.join(output, path))
    I.shutil.copy(hx_path, output)


def my_makedirs(path):
    if not I.os.path.exists(path):
        I.os.makedirs(path)


def copyMaxZFilesToOutput(maxZ_folders, output):
    '''assumes, that max z projections are in folders, which cointain a string like S001, indicating the corresponding slice.
       Further assumes, that the file is precisely named "max_z_projection.tif.
       Copies max z projections to output
       Renames them to foldername + '_max_z_projection.tif' '''
    my_makedirs(output)
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

