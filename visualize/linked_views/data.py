import pandas as pd
from collections.abc import Sequence

class AbstractDataFrameWrapper(object):
    def __init__(self, data):
        self.df = data
        self.name = "Abstract DataFrame"
        self.shape = self.df.shape
        self.columns = self.df.columns

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
    def __init__(self, data):
        super().__init__(data)
        self.columns = data.columns.tolist()

    def binby(self, columns, binsize, mode="mean", value_col=None):
        assert len(columns) == 2, "Please provide an array of 2 columns to bin by. You provided {}".format(columns)
        assert (value_col == None and mode=="count") or (value_col != None and mode != "count"), "If mode = count, you may not pass a value_col, otherwise you must."
        c1, c2 = columns
        if mode == "count":
            return self.df.groupby([(selfdf[c1]/binsize).round()*binsize, 
                    (self.df[c2]/binsize).round()*binsize]).apply(lambda x: x.count())
        else:
            agg = aggregation_mode_mapping[mode] if isinstance(mode, str) else mode
            return self.df.groupby([(selfdf[c1]/binsize).round()*binsize, 
                    (self.df[c2]/binsize).round()*binsize]).apply(lambda x: agg(x[value_col]))

    def count(self, binby=None, shape=None, selection=None, limits=None):
        binsize = get_binsize(shape, limits)
        assert not isinstance(binsize, Sequence), "Nested binning/shape/limits are not supported (yet)."
        if all([x is None for x in [binby, shape, selection, limits]]):
            return self.df.count()
        return self.df.iloc[selection].clip(*limits).binby(columns=binby, binsize=binsize, mode="count")

    def min(self, column=None, binby=None, shape=None, selection=None, limits=None):
        if all([x is None for x in [binby, shape, selection, limits]]):
            if column is None:
                return self.df.min()
            else:
                return self.df[column].min()
        binsize = get_binsize(shape, limits)
        assert not isinstance(binsize, Sequence), "Nested binning/shape/limits are not supported (yet)."
        return self.df.iloc[selection].clip(*limits).binby(columns=binby, binsize=binsize, mode="min")

    def max(self, column=None, binby=None, shape=None, selection=None, limits=None):
        if all([x is None for x in [binby, shape, selection, limits]]):
            if column is None:
                return self.df.max()
            else:
                return self.df[column].max()
        binsize = get_binsize(shape, limits)
        assert not isinstance(binsize, Sequence), "Nested binning/shape/limits are not supported (yet)."
        return self.df.iloc[selection].clip(*limits).binby(columns=binby, binsize=binsize, mode="max")

    def mean(self, column=None, binby=None, shape=None, selection=None, limits=None):
        if all([x is None for x in [binby, shape, selection, limits]]):
            if column is None:
                return self.df.mean()
            else:  
                return self.df[column].mean()
        binsize = get_binsize(shape, limits)
        assert not isinstance(binsize, Sequence), "Nested binning/shape/limits are not supported (yet)."
        return self.df.iloc[selection].clip(*limits).binby(columns=binby, binsize=binsize, mode="mean")

    def median(self, column=None, binby=None, shape=None, selection=None, limits=None):
        if all([x is None for x in [binby, shape, selection, limits]]):
            if column is None:
                return self.df.median()
            else:
                return self.df[column].median()
        binsize = get_binsize(shape, limits)
        assert not isinstance(binsize, Sequence), "Nested binning/shape/limits are not supported (yet)."
        return self.df.iloc[selection].clip(*limits).binby(columns=binby, binsize=binsize, mode="median")
    
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

class VaexTableWrapper(AbstractDataFrameWrapper):
    def __init__(self, data):
        super().__init__(data)
        self.columns = self.df.get_column_names()
    
    def compute_selection(self, ):
        # TODO
        pass

    def calculate(self, operation, *args, **kwargs):
        if not args and "expression" not in kwargs:
            kwargs["expression"] = self.columns

        allowed_operations = ["min", "max", "mean", "median", "count"]
        if operation in allowed_operations:
            operation_func = getattr(self.df, operation)
            return operation_func(*args, **kwargs)
        else:
            raise ValueError(f"Invalid operation. Choose from {allowed_operations}.")

    def min(self, *args, **kwargs):
        return self.calculate("min", *args, **kwargs)

    def max(self, *args, **kwargs):
        return self.calculate("max", *args, **kwargs)

    def mean(self, *args, **kwargs):
        return self.calculate("mean", *args, **kwargs)

    def median(self, *args, **kwargs):
        return self.calculate("median", *args, **kwargs)

    def count(self, *args, **kwargs):
        return self.calculate("count", *args, **kwargs)

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
