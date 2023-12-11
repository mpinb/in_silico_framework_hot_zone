import pandas as pd
from collections.abc import Sequence
import numpy as np

class AbstractDataFrameWrapper(object):
    """
    Baseclass for the DataFrame wrappers.
    This baseclass takes care of everything that the DataFrame wrappers don't have.
    
    All methods or attributes that are not overwritten by specific DataFrame wrappers are passed to the underlying DataFrame instead.

    While it supports dictionary-like __getitem__ and __setitem__ methods, .loc or .iloc are not supported.
    Instead, use the .df attribute to access the underlying DataFrame directly.
    
    """
    def __init__(self, data):
        self.df = data
        self.name = "Abstract DataFrame"

    def __setattr__(self, attr, value):
        if attr in ['df', 'name', 'shape', 'columns']:
            self.__dict__[attr] = value
        else:
            setattr(self.df, attr, value)

    def __setitem__(self, key, value):
        self.df[key] = value

    def __getitem__(self, key):
        return self.df[key]

    def __getattr__(self, attr):
        # I can fetch df from self without infinite recursion
        # because Python first looks if it is in the instance (which it should if it's initialized)
        # before calling __getattr__
        if hasattr(self.df, attr):
            def wrapper(*args, **kwargs):
                """Fetches the correct attr, but doesn't call it yet"""
                method = getattr(self.df, attr)
                return method(*args, **kwargs)
            return wrapper
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}', and neither does its underlying dataframe.")

class PandasTableWrapper(AbstractDataFrameWrapper):
    """
    This class serves as a wrapper around the Pandas DataFrame, providing a unified interface for linked views.

    It overrides the 'min', 'max', 'mean', and 'median' methods to support binning.
    If the arguments ['binby', 'shape', 'selection', 'limits'] are not specified, these methods will operate on the undelrying dataframe with no modifications.
    Otherwise, a vaex-like binning operation will be performed on the underlying dataframe, and the result will be returned.

    Added minmax for vaex-consistent API
    """
    def __init__(self, data):
        super().__init__(data)
        self.columns = data.columns.tolist()

    def binby(self, columns, binsize):
        """Calculates some statistic on a binned representation of the data. 
        These bins can be N-dimensional, but tend to be 1d, 2D and occasionally 3d, as they are used for plotting.

        Args:
            columns (array): Array of column names to bin by.
            binsize (Sequence): binsize defines the size of the bins, and is of shape Nx2, where N is the number of dimensions.

        Returns:
            pd.DataFrameGroupBy: a pd.DataFrameGroupBy object that cannot be used for PandasTableWrapper
        """
        grouped_df = self.df.groupby([(self.df[c]/bs).round()*bs for c, bs in zip(columns, binsize)])
        return grouped_df

    def calc_like_pandas(self, operation, columns=None, inplace=False):
        if columns is None:
            columns = self.columns
        if operation == "minmax":
            filtered_df = self.df[columns].agg(["min", "max"])
        else:
            filtered_df = self.df[columns].agg(operation)
        
        if inplace:
            self.df = filtered_df
            return self
        else:
            return PandasTableWrapper(filtered_df)


    def calculate(self, operation, expression=None, binby=None, shape=None, selection=None, limits=None):
        if all([x is None for x in [binby, shape, selection, limits]]):
            return self.calc_like_pandas(operation, columns=expression, inplace=inplace)

        if limits:
            if not isinstance(limits[0], Sequence):
                limits = [limits]*len(self.columns)
            assert len(limits) == len(binby), "Got {} limits for {} columns".format(len(limits), len(self.columns))
            assert all([len(limit_pair) == 2 for limit_pair in limits]), "All elements in limits should be  pair of values"
            for i, (col, limit) in enumerate(zip(binby, limits)):
                self.df = self.df.loc[(self.df[col] >= limit[0]) & (self.df[col] <= limit[1])]
        
        binsize = get_binsize(shape, limits)
        assert (expression == None and operation=="count") or (expression != None and operation != "count"), "If calculating the count(), you may not pass an expression, otherwise you must."
                    
        self.filtered_df = pandas_table_wrapper.df[selection] if selection is not None else self.df
        grouped_df = self.binby(columns=binby, binsize=binsize)
        if operation == "count":
            r = grouped_df.apply(lambda x: x.count())
        elif operation =="minmax":
            r = grouped_df.apply(lambda x: np.array([x.min(), x.max()]).T)
        else:
            r = getattr(grouped_df[expression], operation)().unstack()
        self.filtered_df = None
        
        return r


    def count(self, *args, **kwargs):
        return self.calculate("count", *args, **kwargs)

    def min(self, *args, **kwargs):
        return self.calculate("min", *args, **kwargs)

    def max(self, *args, **kwargs):
        return self.calculate("max",  *args, **kwargs)

    def mean(self, *args, **kwargs):
        return self.calculate("mean", *args, **kwargs)

    def median(self, *args, **kwargs):
        return self.calculate("median", *args, **kwargs)

    def median_approx(self, *args, **kwargs):
        """Warning: this is same as median for pandas, but not for vaex"""
        return self.calculate("median", *args, **kwargs)
    
    def minmax(self, *args, **kwargs):
        return self.calculate("minmax", *args, **kwargs)
    
    def compute_selection(self, ):
        # TODO
        pass

    def to_dict(self, *args, **kwargs):
        """Returns a dictionary representation of the dataframe.
        Orients by records by default, unless otherwise specified.

        Returns:
            dict: The dictionary
        """
        if "orient" not in kwargs:
            kwargs["orient"] = "records"
        return self.df.to_dict(*args, **kwargs)

    def get_selection(self, indices):
        return self.df.iloc[indices]

class VaexTableWrapper(AbstractDataFrameWrapper):
    """
    This class serves as a wrapper around the Vaex DataFrame, providing a unified interface for linked views. 

    It overrides the 'min', 'max', 'mean', and 'median' methods.
    If the 'expression' argument is not specified, these methods will operate on all columns of the DataFrame by default. 

    This modification allows for more intuitive usage and ensures consistent behavior across different views.
    """
    def __init__(self, data):
        super().__init__(data)
        self.columns = self.df.get_column_names()
        self.shape = self.df.shape
    
    def compute_selection(self, ):
        # TODO
        pass

    def _calculate(self, _operation, *args, **kwargs):
        allowed_operations = ["min", "max", "mean", "median", "count"]
        if _operation in allowed_operations:
            if not args and "expression" not in kwargs:
                kwargs["expression"] = self.columns
            operation_func = getattr(self.df, _operation)
            return operation_func(*args, **kwargs)
        else:
            raise ValueError(f"Invalid operation. Choose from {allowed_operations}.")

    def min(self, *args, **kwargs):
        return self._calculate("min", *args, **kwargs)

    def max(self, *args, **kwargs):
        return self._calculate("max", *args, **kwargs)

    def mean(self, *args, **kwargs):
        return self._calculate("mean", *args, **kwargs)

    def median_approx(self, *args, **kwargs):
        return self._calculate("median_approx", *args, **kwargs)

    def median(self, *args, **kwargs):
        return self._calculate("median", *args, **kwargs)

aggregation_mode_mapping = {
    'mean': lambda x: x.mean,
    'median': lambda x: x.median,
    'min': lambda x: x.min,
    'max': lambda x: x.max,
    'count': lambda x: x.count,
}

def get_binsize_1d(shape, limits):
    binsize = (limits[1] - limits[0])/shape
    return binsize

def get_binsize(shape, limits):
    assert isinstance(shape, Sequence) or isinstance(shape, int), "shape should be an integer or array like"
    if isinstance(shape, int) and isinstance(limits[0], Sequence):
        # limits is nested array implying shape should be array, but shape is scalar
        # vaex supports passing single integer for all columnss
        shape = [shape]*len(limits)
    
    if type(shape) == int:
            return get_binsize_1d(shape, limits)
    else:
        assert len(shape) == len(limits), "Got {} values for shape, but {} limits".format(len(shape, len(limits)))
        assert all([len(limit_pair) == 2 for limit_pair in limits]), "All elements in limits should be  pair of values"
        return [get_binsize_1d(s, l) for s, l in zip(shape, limits)]
