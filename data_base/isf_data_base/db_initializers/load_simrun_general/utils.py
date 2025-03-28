import hashlib, six
import pandas as pd

def _hash_file_content(fn):
    with open(fn, 'rb') as content:
        h = hashlib.md5(content.read()).hexdigest()
    return h


def _get_dumper(value, optimized_pandas_dumper, optimized_dask_dumper):
    """Infer the best dumper for a dataframe.

    Infers the correct parquet dumper for either a pandas or dask dataframe.

    Args:
        value (pd.DataFrame or dd.DataFrame): Dataframe to infer the dumper for.

    Returns:
        module: Dumper module to use for the dataframe.

    Raises:
        NotImplementedError: If the dataframe is not a pandas or dask dataframe.
    """
    if six.PY2:
        # For the legacy py2.7 version, it still uses the msgpack dumper
        from data_base.isf_data_base.IO.LoaderDumper import (
            dask_to_msgpack,
            pandas_to_msgpack,
        )

        return pandas_to_msgpack if isinstance(value, pd.DataFrame) else dask_to_msgpack
    elif six.PY3:
        return (
            optimized_pandas_dumper
            if isinstance(value, pd.DataFrame)
            else optimized_dask_dumper
        )
    else:
        raise NotImplementedError()

        
def convert_df_columns_to_str(df):
    """Convenience method to convert all columns of a dataframe to strings.

    :skip-doc:
    """
    df = df.rename(
        columns={col: "{}".format(col) for col in df.columns if type(col) != str}
    )
    return df
