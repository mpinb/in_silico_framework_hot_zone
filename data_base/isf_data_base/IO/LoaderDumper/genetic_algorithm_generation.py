# In Silico Framework
# Copyright (C) 2025  Max Planck Institute for Neurobiology of Behavior - CAESAR

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# The full license text is also available in the LICENSE file in the root of this repository.
"""Save and load the parameters and objectives of a genetic algorithm to and from numpy ``.npz``.

.. deprecated:: 0.2.0
    This module is deprecated and will be removed in a future version.
    The results of the genetic algorithm are saved in a parquet file using :py:mod:`~data_base.isf_data_base.IO.LoaderDumper.pandas_to_parquet`.
    
See also:
    :py:meth:`biophysics_fitting.optimizer.save_result`
    
:skip-doc:
"""

import os
# import cloudpickle
import compatibility
import numpy as np
from . import parent_classes
import pandas as pd


def check(obj):
    """Check whether the object can be saved with this dumper
    
    Args:
        obj (object): Object to be saved
        
    Returns:
        bool: Whether the object is a tuple of two numpy arrays
    """
    return isinstance(obj[0], np) and isinstance(obj[1], np)


class Loader(parent_classes.Loader):
    """Loader for genetic algorithm generations
    
    Attributes:
        generation (int): Generation number
        parameters (list): List of parameter names
        objectives (list): List of objective names
    """
    def __init__(self, generation=None, parameters=None, objectives=None):
        self.generation = generation
        self.parameters = parameters
        self.objectives = objectives

    def get(self, savedir):
        parameters = np.load(os.path.join(savedir, 'parameters.npz'))['arr_0']
        objectives = np.load(os.path.join(savedir, 'objectives.npz'))['arr_0']
        pdf_parameters = pd.DataFrame(parameters, columns=self.parameters)
        pdf_objectives = pd.DataFrame(objectives, columns=self.objectives)
        return pd.concat([pdf_objectives, pdf_parameters], axis=1)


def dump(obj, savedir, generation, parameters, objectives):
    np.savez_compressed(os.path.join(savedir, 'parameters.npz'), arr_0=obj[0])
    np.savez_compressed(os.path.join(savedir, 'objectives.npz'), arr_0=obj[1])

    #     with open(os.path.join(savedir, 'Loader.pickle'), 'wb') as file_:
    #         cloudpickle.dump(Loader(generation, parameters, objectives), file_, generation, parameters)
    compatibility.cloudpickle_fun(
        Loader(generation, parameters, objectives),
        os.path.join(savedir, 'Loader.pickle'))
