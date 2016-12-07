import dask.dataframe
import pandas as pd
import os

def read_cell_activation_times(mdb):
    out = []
    for index, row in mdb.metadata.iterrows():
        absolute_path_to_file = os.path.join(mdb.tempdir, row.path, row.cell_file_name)
        out.append(dask.delayed(pd.read_csv)(absolute_path_to_file))
    return dask.dataframe.from_delayed(out)