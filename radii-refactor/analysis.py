import re
import os
import pandas as pd
import radii as radi
import transformTools as tr


def get_slice_number_from_path(path):
    path = os.path.basename(path)
    out = re.findall(r'[sS]\d+', path)

    if len(out) != 1:
        print("there are two slice names indicator exist in the name of the file below:")
        print(out)
    return out[0]


def test_get_slice_number_from_path():
    out = get_slice_number_from_path('S02')
    assert (isinstance(out, str))
    out = get_slice_number_from_path('S03/S02')
    assert (out == 'S02')
    try:
        get_slice_number_from_path('S02S02')
        raise ValueError('Expected AssertionError!')
    except AssertionError:
        pass


# test_get_slice_number_from_path()
# get_slice_number_from_path('S02')

def get_path_by_slice_number_dict(pathlist):
    lookup_dict = {int(get_slice_number_from_path(path)[1:]): path for path in pathlist}
    return lookup_dict


## Measurment are in micrometer
def findNeighbours(points, width=2):
    listOfNeighbours = []
    for idx, p_i in enumerate(points):
        neighbours = []
        neighbours.append(p_i[:3])
        for idy, p_j in enumerate(points):
            if (idx != idy and p_i[0] // width == p_j[0] // width and p_i[1] // width == p_j[1] // width):
                neighbours.append(p_j[:3])
        listOfNeighbours.append(neighbours)
    return listOfNeighbours


def get_path_by_slice_number_dict(pathlist):
    lookup_dict = {int(get_slice_number_from_path(path)[1:]): path for path in pathlist}
    return lookup_dict


def compare_points(p1, p2):
    assert (len(p1) == len(p2))
    return max([np.abs(pp1 - pp2) for pp1, pp2 in zip(p1, p2)])


def searchForPointInList(main_point, radii_details):
    #    print "in searchForPointInList(), the main point is:", main_point
    #    print "in searchForPointInList(), radii_details is:" , radii_details
    diffs = []
    for idx, p in enumerate(radii_details["Inten. orig points"]):
        point = p[0]
        diff = compare_points(point, main_point)
        diffs.append(diff)
        if diff <= 10E-14:
            #            print "in searchForPointInList(), if diff == 0, the idx is:", idx
            #            print "in searchForPointInList(), if diff == 0, the radii_details['Ray points'][idx] is:", radii_details["Ray points"][idx]
            return [idx, radii_details["Ray points"][idx]]
    print 'min diff', min(diffs), point, main_point


def makeContourFromRays(rays):
    return [p for ray in rays for p in ray if p != []]


def slope(p1, p2):
    if (p1[0] - p2[0] == 0):
        return 10E+10
    return (p1[1] - p2[1]) / (p1[0] - p2[0])


def intercept(m, p2):
    return -m * p2[0] + p2[1]


## Finding overlapes between points by looking at their rays
## slope is m
def constructPolygonFromContour(contour):
    lines = []
    for i, p1 in enumerate(contour):
        p2 = contour[i + 1] if i + 1 != len(contour) else contour[0]
        #        print "in constructPolygonFromContour(), in for loop, p1, p2 are:",p1, p2
        m = slope(p1, p2)
        #        print "in constructPolygonFromContour(), in for loop, p1, p2 are:",p1, p2
        b = intercept(m, p2)
        line = [p1, p2, m, b]
        lines.append(line)
    return lines


def findIntersection(line1, line2):
    if line1[2] - line2[2] == 0:
        return [10E+10, 10E+10]
    x = (line2[3] - line1[3]) / (line1[2] - line2[2])
    y = line1[2] * x + line1[3]
    return [x, y]


def checkOverlaps(contour1, contour2):
    lines1 = constructPolygonFromContour(contour1)
    lines2 = constructPolygonFromContour(contour2)
    for line1 in lines1:
        for line2 in lines2:
            ints = findIntersection(line1, line2)
            if (line1[0][0] <= ints[0] <= line1[1][0] and line1[0][1] <= ints[1] <= line1[1][1]):
                return True


def findOverlaps(points, listOfNeighbours, radii_Details):
    overlaps = []
    for nr_points in listOfNeighbours:
        main_point = nr_points[0]
        #        print "in findOverlap(), the radiiDetails is:", radii_Details

        neighbours = nr_points[1:]
        if (neighbours == []): continue
        temp = searchForPointInList(main_point, radii_Details)
        inedxOfmainPoint = temp[0]
        main_point_rays = temp[1]
        #        print "in findOverlaps(), main_point_rays: is", main_point_rays
        for neighbour_point in neighbours:
            temp = searchForPointInList(neighbour_point, radii_Details)

            inedxOfNeighbour = temp[0]
            neighbour_point_rays = temp[1]

            #            print "in findOverlaps(), in for loop, neighbour_rays is:", neighbour_point_rays
            main_point_contour = makeContourFromRays(main_point_rays)
            neighbour_point_contour = makeContourFromRays(neighbour_point_rays)
            if checkOverlaps(main_point_contour, neighbour_point_contour):
                if (compare_points(main_point, neighbour_point) >= 10E-14):
                    overlaps.append([main_point, neighbour_point])

    return overlaps


def test_findOverlap():
    def get_am_files(dir_):
        list_ = [os.path.join(dir_, amFile) for amFile in os.listdir(dir_) if amFile.endswith(".am")]
        return get_path_by_slice_number_dict(list_)

    slice_number = 6
    am050Paths = get_am_files(rad.amOutput050)
    am050_pointsWithRad = tr.read.amFile(am050Paths[slice_number])
    #    print(am050Paths[slice_number])
    am050_points = [p[:3] for p in am050_pointsWithRad]

    listOfNeighbours = findNeighbours(am050_points)
    #    print  "in test_findOverlap(), listOfNeighbours is:", listOfNeighbours[0]
    radiiDetails_all = {v['Slice name']: v for v in radiiDatailsList if v['Treshold'] == 0.50}
    #    print "in test_findOverlap(), the radiiDetails_all[6] is:", radiiDetails_all[6]

    overlaps = findOverlaps(am050_points, listOfNeighbours, radiiDetails_all[6])

    return overlaps


overlaps = test_findOverlap()
print overlaps


def get_all_data_output_table(all_slices):
    for key in sorted(all_slices.keys()):
        slice_object = all_slices[key]
        slice_object.


def allData(am050_tr_folder, radi_object, radiiDetails):
    # the amPoints_hoc contains the transfromed am points to the hoc coordinates
    # and it is also contains the extracted radi for tht specific points
    # the am_hoc_pairs is actually the pairs of am and hoc points that
    # chose to add the radi from the am to that specific hoc point.
    # the am_hoc_pairs contains the pair of points like this:
    # [[[am point], [hoc point]], ... ]
    # reading extracted radii for the tresholds 025, 050, 075
    # from their corresponding folder and files, and saving them in arrays again.

    # radiiDetails contains inforamtion of intenisties for points, post_points, and it also include the
    # contour of around a point that we extract the radius of that point from it. The shape of one element of this list
    # is like below:
    # ["sliceName": "some name", "Treshold": a float number, "Inten. orig points": [[x, y ,z, inten.], ...],
    # "Inten. post points": [[x_pm, y_pm ,z_pm, inten.], ...], "Ray points":
    # [
    # [[[x0_back, y0_back],[x0_front, y0_front]], [[x1_back, y1_back],[x1_front, y1_front]], ... ], --> rays of the first point
    # [[[x0_back, y0_back],[x0_front, y0_front]], [[x1_back, y1_back],[x1_front, y1_front]], ... ], --> rays of the second point
    # ...
    # ]

    # reading transformaed coordinates from am050_tr_folder

    def get_am_files(dir_):
        list_ = [os.path.join(dir_, amFile) for amFile in os.listdir(dir_) if amFile.endswith(".am")]
        return get_path_by_slice_number_dict(list_)

    am025Paths = get_am_files(radi_object.amOutput025)
    am050Paths = get_am_files(radi_object.amOutput050)
    am075Paths = get_am_files(radi_object.amOutput075)
    amTrPaths = get_am_files(am050_tr_folder)

    maxZPathList = radi_object.maxZPathList
    am_hoc_pairs = radi_object.am_hoc_pairs

    dfs = []
    df_c = pd.DataFrame()
    for slice_number in am050Paths.keys():  # idx in range(file_numbers):
        ###############################
        # construct column(s) containing points in the coordinate system of individual slices
        ###############################
        am025_pointsWithRad = tr.read.amFile(am025Paths[slice_number])
        am050_pointsWithRad = tr.read.amFile(am050Paths[slice_number])
        am075_pointsWithRad = tr.read.amFile(am075Paths[slice_number])
        # extract radii
        rads_050 = [p[3] for p in am050_pointsWithRad]
        rads_075 = [p[3] for p in am075_pointsWithRad]
        # putting the data from one slice into a temporary data frame
        df_individual_slice = pd.DataFrame(am025_pointsWithRad, columns=["x", "y", "z", "radius 025"])
        # adding the slice name to all points of the temporary data frame of one slice
        df_individual_slice.insert(0, "slice", slice_number)
        df_individual_slice_no_dups = df_individual_slice
        # adding the different treshold of radii of the slice to the temporary data frame
        df_individual_slice["radius 050"] = rads_050
        df_individual_slice["radius 075"] = rads_075

        ###############################
        # construct column(s) containing points in the coordinate system of aligned slices
        ###############################
        amTr_pointsWithRad = tr.read.amFile(amTrPaths[slice_number])
        amTr_points = [p[0:3] for p in amTr_pointsWithRad]  # this will become the hx column

        df_aligned_slices = pd.DataFrame(amTr_points, columns=["x_hx", "y_hx", "z_hx"])

        ###############################
        # construct column(s) containing points in the coordinate system of the hoc file
        ###############################
        amPoints_in_hoc = tr.exTrMatrix.applyTransformationMatrix(amTr_points, radi_object.trMatrix)
        df_hoc = pd.DataFrame(amPoints_in_hoc, columns=["x_hoc", "y_hoc", "z_hoc"])

        ###############################
        # add intensities and Ray points
        ##############################

        df_intensities = I.pd.DataFrame(radiiDetails[slice_number])
        df_i = df_intensities
        df_intensities = df_i.apply(lambda x: I.pd.Series(x['Inten. orig points'][0] + [x['Inten. orig points'][1]],
                                                          index=['x_prem', 'y_prem', 'z_prem', 'intensity']), axis=1)

        df_intensities_pm = df_i.apply(lambda x: I.pd.Series(x['Inten. post points'][0] + [x['Inten. post points'][1]],
                                                             index=['x_pm', 'y_pm', 'z_pm', 'intensity_pm']), axis=1)

        ###############################
        # add intensities and ray Points
        ##############################
        df_r = I.pd.DataFrame(radiiDetails[slice_number])
        df_rayPoints = I.pd.DataFrame(df_r['Ray points'])
        #       df_rayPoints = df_r.apply(lambda x: I.pd.Series(x['Ray points'], index = ['Ray points Treshold=0.50']), axis = 1)

        ############################
        # find neighbours
        ############################
        listOfNeighbours = findNeighbours(am050_pointsWithRad)
        df_tn = I.pd.DataFrame(listOfNeighbours, columns=["original_point", "neihgbours"])

        ############################
        # find overlapes of neighbours
        ############################

        findOverlaps(listOfNeighbours, radiiDetails[slice_number])

        ###############################
        # construct full dataframe of this iteration of the for loop
        ###############################

        df_temp = pd.concat([df_individual_slice, df_aligned_slices, df_hoc, df_intensities,
                             df_intensities_pm, df_rayPoints, df_tn], axis=1)
        #        df_temp = pd.concat([df_individual_slice, df_aligned_slices, df_hoc ], axis = 1)

        # append the temporary data frame to the whole cell data frame
        dfs.append(df_temp)

    df = pd.concat(dfs)

    # get rid of duplicated points in the amira files
    # df = df.drop_duplicates(subset = ['x', 'y', 'z', 'slice'])
    ##################################
    # add aligned hoc point column
    ##################################
    hoc_pairs_dict_ = I.defaultdict(lambda: list())
    for am_coord, hoc_cord in am_hoc_pairs:
        hoc_pairs_dict_[tuple(am_coord[:3])].append(tuple(hoc_cord[:3]))
    df['aligned_hoc_point'] = df.apply(lambda x: hoc_pairs_dict_[(x['x_hoc'], x['y_hoc'], x['z_hoc'])], axis=1)

    ###############################
    # construct column(s) from radiiDetails
    ###############################

    # df_radiiDetails = pd.DataFrame(radiiDetailsList)

    return df


def dataAnalysis(self, amTrFolder, radiiDetailsList):
    return allData(amTrFolder, self, radiiDetailsList)


df = dataAnalysis(rad, am050_tr_folder, {v['Slice name']: v for v in radiiDatailsList if v['Treshold'] == 0.50})
# df.loc[lambda dfs: dfs['Treshold'] == 0.25]
