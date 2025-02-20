'''Modify the cell and/or network after both have been initalized.

Such a function can for example be used to deactivate specific synapses at a soma distance.
'''
import importlib

__author__ = "Arco Bast"
__date__ = "2019-02-16"

def get(funname):
    '''Get the function with the given name.

    Network modify functions reside in a module of the same name.
    This method fetches them from said module.
    
    Args:
        funname (str): Name of the function to get.

    Returns:
        callable: The function with the given name.
    '''
    module = importlib.import_module(__name__ + '.' + funname)
    fun = getattr(module, funname)
    return fun
