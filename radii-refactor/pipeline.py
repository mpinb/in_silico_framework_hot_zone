import thickness as th
import transformation as tr
import IO
import utils as u
import analysis as an


class SliceData:
    def __init__(self, slice_name=None, slice_threshold=None):
        self.slice_name = slice_name
        self.am_file_path = None
        self.image_file_path = None
        self.output_path = None
        self.am_object = None
        self.points = None
        self.transformed_points = None
        self.slice_thicknesses_object = None
        self.transformation_object = None
        self.slice_threshold = slice_threshold

    def set_am_file_path(self, path):
        self.am_file_path = path

    def set_image_file_path(self, path):
        self.image_file_path = path

    def set_output_path(self, path):
        self.output_path = path

    def write_output(self):
        self.am_object.write()

    def set_slice_name(self, slice_name=None):
        if slice_name is not None:
            self.slice_name = slice_name
        else:
            self.slice_name = u.get_slice_name(self.am_file_path, self.image_file_path)

    # TODO: provide default values
    def compute(self, xy_resolution, z_resolution,
                ray_length_front_to_back_in_micron,
                number_of_rays, threshold_percentage,
                max_seed_correction_radius_in_micron):
        self.am_object = IO.Am(self.am_file_path, self.output_path)
        self.am_object.read()
        self.points = self.am_object.all_data["POINT { float[3] EdgePointCoordinates }"]
        slice_thicknesses_object = th.Thickness_extractor(self.points, self.image_file_path,
                                                          xy_resolution, z_resolution,
                                                          ray_length_front_to_back_in_micron,
                                                          number_of_rays, threshold_percentage,
                                                          max_seed_correction_radius_in_micron)
        self.slice_thicknesses_object = slice_thicknesses_object


class ExtractThicknessPipeline:
    def __init__(self):
        # ---- files and folders
        self.hoc_file = None
        self.hoc_object = None
        self.am_paths = []
        self.tif_paths = []
        self._3D = False
        self.output_folder = None
        # --- thickness extractor class parameters:
        self.default_threshold = 0.5
        self.thresholds_list = [self.default_threshold]
        self.xy_resolution = None
        self.z_resolution = None
        self.number_of_rays = None
        self.ray_length_front_to_back_in_micron = None
        self.max_seed_correction_radius_in_micron = None
        # --- objects and dicts
        self.am_tif = {}
        self.all_slices = {}
        self.all_points_with_default_threshold = []
        self.all_transformed_points_with_default_threshold = []
        self.all_thicknesses = {}

        # --- transformation
        # bijective_points are set/list/dict of 4 points from
        self.bijective_points = None

    def set_am_paths_by_list(self, am_paths_list):
        self.am_paths = am_paths_list

    def set_am_paths_by_folder(self, folder_path_to_am_files):
        self.am_paths = u.get_files_by_folder(folder_path_to_am_files, file_extension="am")

    def set_am_paths_by_hx(self, folder_path_to_hx_file):
        self.am_paths = u.get_am_paths_from_hx(folder_path_to_hx_file)

    def set_output_path(self, path_to_output_folder):
        self.output_folder = path_to_output_folder
        if self.hoc_object is not None:
            self.hoc_object.output_path = self.output_folder

    def set_hoc_file(self, path_to_hoc_file):
        self.hoc_file = path_to_hoc_file
        if self.output_folder is None:
            raise RuntimeError("Please first set_output_path")
        else:
            self.hoc_object = IO.Hoc(self.hoc_file,
                                     self.output_folder + "/"
                                     + u.get_file_name_from_path(self.hoc_file))

    def set_tif_paths_by_list(self, tif_paths_list):
        self.tif_paths = tif_paths_list

    def set_tif_paths_by_folder(self, folder_path_to_tif_files):
        self.tif_paths = u.get_files_by_folder(folder_path_to_tif_files, file_extension="tif")

    def set_3d_tif_stack_by_folder(self, folder_path_to_3d_tif_files):
        self.tif_paths = u.get_files_by_folder(folder_path_to_3d_tif_files, file_extension="tif")
        self._3D = True

    # TODO: EG: to put in init with defaults, set_thresholds, set_thickness..
    def set_thresholds(self, thresholds_list):
        """
        :param thresholds_list: A list contains thresholds values for extracting thicknesses
        :return: A setter function, No return, if not set the default value of 0.5 will be used
        """
        self.thresholds_list = thresholds_list

    def set_thickness_extractor_parameters(self, xy_resolution=0.092,
                                           z_resolution=0.5, number_of_rays=36,
                                           ray_length_front_to_back_in_micron=20,
                                           max_seed_correction_radius_in_micron=10):
        self.xy_resolution = xy_resolution
        self.z_resolution = z_resolution
        self.number_of_rays = number_of_rays
        self.ray_length_front_to_back_in_micron = ray_length_front_to_back_in_micron
        self.max_seed_correction_radius_in_micron = max_seed_correction_radius_in_micron

    def set_bijective_points_by_list(self, bi_points):
        self.bijective_points = bi_points

    def set_bijective_points_by_file(self, input_path):
        self.bijective_points = IO.read_landmark_file(input_path)

    def set_bijective_points_automatically(self):
        """ TODO: This function will call the poor function that we developed and not working well"""
        pass

    def run(self):
        self._initialize_project()
        self._extract_thicknesses()
        self._transform_points()
        self._update_hoc_file_with_thicknesses()
        self._compute_all_data_table()

    def _initialize_project(self):

        if self.output_folder is None:
            self.output_folder = u.make_directories(self.output_folder)
        for threshold in self.thresholds_list:
            u.make_directories(self.output_folder + "/" + str(threshold))

    def _extract_thicknesses(self):
        thresholds = self.thresholds_list
        if self.am_paths is None or self.tif_paths is None:
            raise RuntimeError("You need to set am and tif paths")
        for threshold in thresholds:
            all_slices_in_threshold = {}
            for am_file in self.am_paths:
                slice_object = SliceData(slice_threshold=threshold)
                slice_object.set_am_file_path(am_file)
                slice_object.set_image_file_path(u.get_am_image_match([am_file],
                                                                      self.tif_paths)[am_file])
                slice_object.set_slice_name()
                slice_object.set_output_path(self.output_folder + "/" + str(threshold) +
                                             "/" + u.get_file_name_from_path(am_file))
                s = self
                slice_object.compute(s.xy_resolution, s.z_resolution,
                                     s.ray_length_front_to_back_in_micron,
                                     s.number_of_rays, threshold,
                                     s.max_seed_correction_radius_in_micron)
                slice_object.write_output()
                all_slices_in_threshold[slice_object] = slice_object
            self.all_slices[threshold] = all_slices_in_threshold
            # self.all_slices[(slice_object.slice_name, slice_object.slice_threshold)] = slice_object

    def _transform_points(self):
        transformation_object = tr.AffineTransformation()
        print self.bijective_points
        transformation_object.set_transformation_matrix_by_aligned_points(self.bijective_points[:3],
                                                                          self.bijective_points[3:])
        for threshold in self.thresholds_list:
            all_slices_with_same_threshold = self.all_slices[threshold]
            for key in sorted(all_slices_with_same_threshold.keys()):
                slice_object = all_slices_with_same_threshold[key]
                transformation_object.transformed_points(slice_object.points, forwards=True)
                slice_object.transformed_points = transformation_object.transformed_points
                slice_object.transformation_object = transformation_object

    def _update_hoc_file_with_thicknesses(self):
        for hoc_point in self.hoc_object.all_data["points"]:
            nearest_point_index = self.all_transformed_points_with_default_threshold.index(
                u.get_nearest_point(hoc_point, self.all_transformed_points_with_default_threshold)
            )
            self.hoc_object.all_data["thicknesses"].append(
                self.all_thicknesses[self.default_threshold][nearest_point_index]
            )

        self.hoc_object.update_thicknesses()

    def _compute_all_data_table(self):
        return an.get_all_data_output_table(self.all_slices, self.default_threshold)

    def _stacking_all_slices(self):
        all_slices_with_default_threshold = self.all_slices[self.default_threshold]
        self.all_points_with_default_threshold = [point for key in sorted(all_slices_with_default_threshold.keys())
                                                  for point in all_slices_with_default_threshold[key].points]

        self.all_transformed_points_with_default_threshold = [tr_points for key in
                                                              sorted(all_slices_with_default_threshold.keys())
                                                              for tr_points in
                                                              all_slices_with_default_threshold[key].transformed_points]
        for threshold in self.thresholds_list:
            all_slices_with_same_threshold = self.all_slices[threshold]
            self.all_thicknesses = {
                threshold: [all_slices_with_same_threshold[key].slice_thicknesses_object[point]["min_thickness"] for key
                            in sorted(all_slices_with_same_threshold.keys()) for point in
                            all_slices_with_same_threshold[key].points]}

        assert len(self.all_points_with_default_threshold) == len(
            self.all_transformed_points_with_default_threshold) == len(
            self.all_thicknesses.values()[0])
        assert len(self.all_thicknesses) == len([1 for thicknesses_in_threshold in self.all_thicknesses.values()
                                                 if len(thicknesses_in_threshold) ==
                                                 len(self.all_thicknesses.values()[0])])
