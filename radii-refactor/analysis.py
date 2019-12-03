import pandas as pd


def get_all_data_output_table(all_slices, default_thresholds):
    df_all_slices = pd.DataFrame()
    for threshold in sorted(all_slices.keys()):
        all_slices_with_same_threshold = all_slices[threshold]
        df_individual_slice = pd.DataFrame()
        for key in sorted(all_slices_with_same_threshold.keys()):
            slice_object = all_slices_with_same_threshold[key]
            if threshold == default_thresholds:
                df_individual_slice_name = pd.DataFrame(slice_object.slice_name, "slice_name")
                df_individual_slice_points = pd.DataFrame(slice_object.points, columns=["x", "y", "z"])
                df_individual_slice_tr_points = pd.DataFrame(slice_object.transformed_points,
                                                             columns=["x_tr", "y_tr", "z_tr"])
                df_individual_slice = pd.concat([df_individual_slice_name,
                                                 df_individual_slice_points,
                                                 df_individual_slice_tr_points,
                                                 ], axis=1)
            df_individual_slice_thicknesses_object_all_data = pd.DataFrame(
                slice_object.slice_thicknesses_object.all.data)
            df_individual_slice = pd.concat([df_individual_slice,
                                             df_individual_slice_thicknesses_object_all_data,
                                             ], axis=1)
        df_all_slices = pd.concat([df_all_slices, df_individual_slice], axis=0)

    return df_all_slices