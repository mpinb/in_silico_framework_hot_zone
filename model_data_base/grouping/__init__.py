# import dask.dataframe as dd
# import pandas as pd
# import group_dataframe_by_another.groupybyanother
# from types import MethodType
# 
# def rebinder(f):
#     if not isinstance(f,MethodType):
#         raise TypeError("rebinder was intended for rebinding methods")
#     def wrapper(*args,**kw):
#         return f(*args,**kw)
#     return wrapper
# 
# 
# pd.DataFrame.groupbyanother = group_dataframe_by_another.groupybyanother