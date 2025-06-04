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
"""
This module provides method to set up the parameters for a Layer 5 Pyramidal Tract neuron (L5PT/L5tt)

These parameters and templates are used to set up the biophysical constraints for the L5PT cell in e.g. :py:mod:`~biophysics_fitting.simulator`.
"""


#########################################
# naming converters scp <--> hay
#########################################
import six
from single_cell_parser.parameters import ParameterSet
import pandas as pd
import logging
logger = logging.getLogger("ISF").getChild(__name__)

def hay_param_to_scp_neuron_param(p):
    """Convert a Hay parameter name to a SCP neuron parameter name.

    This function is a simple converter that converts the Hay naming convention to what ISF
    uses internally in :py:mod:`~single_cell_parser.cell.Cell`
    
    Args:
        p (str): The Hay parameter name.
        
    Returns:
        str: The SCP neuron parameter name.

    Example:

        >>> hay_param_name = "NaTa_t.axon.gNaTa_tbar"
        >>> hay_param_to_scp_neuron_param(hay_param_name)
        "NaTa_t.AIS.gNaTa_tbar"  # axon --> AIS
        
    See also:
        See :cite:t:`Hay_Hill_Schuermann_Markram_Segev_2011` for more information.
    """
    p = p.split('.')
    if p[1] == 'axon':
        p[1] = 'AIS'
    elif p[1] == 'apic':
        p[1] = 'ApicalDendrite'
    elif p[1] == 'dend':
        p[1] = 'Dendrite'
    elif p[1] == 'soma':
        p[1] = 'Soma'
    if p[0] == 'none' and p[2] == 'g_pas':
        p[0] = 'pas'
        p[2] = 'g'
    return '.'.join([p[1], 'mechanisms', 'range', p[0], p[2]])


def hay_params_to_scp_neuron_params(params):
    """Convert a list of Hay parameter names to a list of SCP neuron parameter names.
    
    Args:
        params (list): The list of Hay parameter names.
        
    Returns:
        list: The list of SCP neuron parameter names."""
    return [hay_param_to_scp_neuron_param(p) for p in params]


#############################################
# template and template modify functions
#############################################


def get_L5tt_template():
    """Get a template cell parameter dictionary for a L5PT cell.
    
    This method returns a nested dictionary-like object that can be used to set up a L5PT cell for simulations.
    The values of each key are set to None or default values, and need to be filled in with the actual values.
    This dictionary-like parameter structure is used by e.g. the :py:class:`~biophysics_fitting.simulator.Simulator` object.
    It provides information on:
    
    - For each section label (for an L5PT: Soma, AIS, ApicalDendrite, Dendrite, Myelin):
        - ``neuron.<section_label>.mechanisms``: active biophysics of the cell (e.g. ion channel densities)
        - ``neuron.<section_label>.properties``: passive biophysics of the cell (e.g. membrane capacitance)
    - sim: simulation parameters:
        - ``T``: temperature
        - ``Vinit``: initial voltage
        - ``dt``: time step
        - ``recordingSites``: recording sites
        - ``tStart``: start time
        - ``tStop``: stop time
            
    Returns:
        :py:class:`~single_cell_parser.parameters.ParameterSet`: The template cell parameters.       
    
    """
    p = {
        'NMODL_mechanisms': {
            'channels': '/'
        },
        'info': {
            'author': 'regger and abast',
            'date': '2018',
            'name': 'scp_optimizer'
        },
        'mech_globals': {},
        'neuron': {
            'AIS': {
                'mechanisms': {
                    'global': {},
                    'range': {
                        'CaDynamics_E2': {
                            'decay': None,
                            'gamma': None,
                            'spatial': 'uniform'
                        },
                        'Ca_HVA': {
                            'gCa_HVAbar': None,
                            'spatial': 'uniform'
                        },
                        'Ca_LVAst': {
                            'gCa_LVAstbar': None,
                            'spatial': 'uniform'
                        },
                        'Ih': {
                            'gIhbar': 8e-05,
                            'spatial': 'uniform'
                        },
                        'K_Pst': {
                            'gK_Pstbar': None,
                            'spatial': 'uniform'
                        },
                        'K_Tst': {
                            'gK_Tstbar': None,
                            'spatial': 'uniform'
                        },
                        'NaTa_t': {
                            'gNaTa_tbar': None,
                            'spatial': 'uniform'
                        },
                        'Nap_Et2': {
                            'gNap_Et2bar': None,
                            'spatial': 'uniform'
                        },
                        'SK_E2': {
                            'gSK_E2bar': None,
                            'spatial': 'uniform'
                        },
                        'SKv3_1': {
                            'gSKv3_1bar': None,
                            'spatial': 'uniform'
                        },
                        'pas': {
                            'e': -90,
                            'g': None,
                            'spatial': 'uniform'
                        }
                    }
                },
                'properties': {
                    'Ra': 100.0,
                    'cm': 1.0,
                    'ions': {
                        'ek': -85.0,
                        'ena': 50.0
                    }
                }
            },
            'ApicalDendrite': {
                'mechanisms': {
                    'global': {},
                    'range': {
                        'CaDynamics_E2': {
                            'decay': None,
                            'gamma': None,
                            'spatial': 'uniform'
                        },
                        'Ca_HVA': {
                            'begin': 900.0,
                            'end': 1100.0,
                            'gCa_HVAbar': None,
                            'outsidescale': 0.1,
                            'spatial': 'uniform_range'
                        },
                        'Ca_LVAst': {
                            'begin': 900.0,
                            'end': 1100.0,
                            'gCa_LVAstbar': None,
                            'outsidescale': 0.01,
                            'spatial': 'uniform_range'
                        },
                        'Ih': {
                            '_lambda': 3.6161,
                            'distance': 'relative',
                            'gIhbar': 0.0002,
                            'linScale': 2.087,
                            'offset': -0.8696,
                            'spatial': 'exponential',
                            'xOffset': 0.0
                        },
                        'Im': {
                            'gImbar': None,
                            'spatial': 'uniform'
                        },
                        'NaTa_t': {
                            'gNaTa_tbar': None,
                            'spatial': 'uniform'
                        },
                        'SK_E2': {
                            'gSK_E2bar': None,
                            'spatial': 'uniform'
                        },
                        'SKv3_1': {
                            'gSKv3_1bar': None,
                            'spatial': 'uniform'
                        },
                        'pas': {
                            'e': -90,
                            'g': None,
                            'spatial': 'uniform'
                        }
                    }
                },
                'properties': {
                    'Ra': 100.0,
                    'cm': 2.0,
                    'ions': {
                        'ek': -85.0,
                        'ena': 50.0
                    }
                }
            },
            'Dendrite': {
                'mechanisms': {
                    'global': {},
                    'range': {
                        'Ih': {
                            'gIhbar': 0.0002,
                            'spatial': 'uniform'
                        },
                        'pas': {
                            'e': -90.0,
                            'g': None,
                            'spatial': 'uniform'
                        }
                    }
                },
                'properties': {
                    'Ra': 100.0,
                    'cm': 2.0
                }
            },
            'Myelin': {
                'mechanisms': {
                    'global': {},
                    'range': {
                        'pas': {
                            'e': -90.0,
                            'g': 4e-05,
                            'spatial': 'uniform'
                        }
                    }
                },
                'properties': {
                    'Ra': 100.0,
                    'cm': 0.02
                }
            },
            'Soma': {
                'mechanisms': {
                    'global': {},
                    'range': {
                        'CaDynamics_E2': {
                            'decay': None,
                            'gamma': None,
                            'spatial': 'uniform'
                        },
                        'Ca_HVA': {
                            'gCa_HVAbar': None,
                            'spatial': 'uniform'
                        },
                        'Ca_LVAst': {
                            'gCa_LVAstbar': None,
                            'spatial': 'uniform'
                        },
                        'Ih': {
                            'gIhbar': 8e-05,
                            'spatial': 'uniform'
                        },
                        'K_Pst': {
                            'gK_Pstbar': None,
                            'spatial': 'uniform'
                        },
                        'K_Tst': {
                            'gK_Tstbar': None,
                            'spatial': 'uniform'
                        },
                        'NaTa_t': {
                            'gNaTa_tbar': None,
                            'spatial': 'uniform'
                        },
                        'Nap_Et2': {
                            'gNap_Et2bar': None,
                            'spatial': 'uniform'
                        },
                        'SK_E2': {
                            'gSK_E2bar': None,
                            'spatial': 'uniform'
                        },
                        'SKv3_1': {
                            'gSKv3_1bar': None,
                            'spatial': 'uniform'
                        },
                        'pas': {
                            'e': -90,
                            'g': None,
                            'spatial': 'uniform'
                        }
                    }
                },
                'properties': {
                    'Ra': 100.0,
                    'cm': 1.0,
                    'ions': {
                        'ek': -85.0,
                        'ena': 50.0
                    }
                }
            },
            'filename': None
        },
        'sim': {
            'T': 34.0,
            'Vinit': -75.0,
            'dt': 0.025,
            'recordingSites': [],
            'tStart': 0.0,
            'tStop': 300.0
        }
    }
    return ParameterSet(p['neuron'])

def get_L5tt_template_v2():
    """Get a template cell parameter dictionary for a L5PT cell.
    
    This method is identical to :py:meth:`get_L5tt_template`, but adds the following specifications:
    
    - The CaDynamics_E2 mechanism is replaced with CaDynamics_E2_v2 (see :py:mod:`mechanisms`).
    - The SKv3_1 mechanism is set to have a linear spatial distribution with intercept (see :cite:t:`Schaefer_Helmstaedter_Schmitt_Bar_Yehuda_Almog_Ben_Porat_Sakmann_Korngreen_2007`).
        
    Returns:
        :py:class:`:py:class:`~single_cell_parser.parameters.ParameterSet``: The template cell parameters.
    """
    neup = get_L5tt_template()
    for loc in neup:
        if loc == 'filename':
            continue
        mechanisms = neup[loc]['mechanisms']['range']
        if 'CaDynamics_E2' in mechanisms:
            mechanisms['CaDynamics_E2_v2'] = mechanisms['CaDynamics_E2']
            del neup[loc]['mechanisms']['range']['CaDynamics_E2']
    apic_skv31 = neup['ApicalDendrite']['mechanisms']['range']['SKv3_1']
    apic_skv31['offset'] = None
    apic_skv31['slope'] = None
    apic_skv31['spatial'] = 'linear'
    apic_skv31['distance'] = 'relative'
    neup['cell_modify_functions'] = {'scale_apical': {'scale': None}}
            
    p = {
        'NMODL_mechanisms': {
            'channels': '/'
        },
        'info': {
            'author': 'regger and abast',
            'date': '2018',
            'name': 'scp_optimizer'
        },
        'mech_globals': {},
        'neuron': neup}
    return ParameterSet(p['neuron'])


def set_morphology(cell_param, filename=None):
    """Add the morphology to a cell parameter object.
    
    The morphology is simply a path to a :ref:`hoc_file_format` file in string format.
    
    Args:
        cell_param (:py:class:`~single_cell_parser.parameters.ParameterSet` | dict): The cell parameter dictionary.
        filename (str): The path to the :ref:`hoc_file_format` file.
        
    Returns:
        :py:class:`~single_cell_parser.parameters.ParameterSet` | dict: The updated cell parameter dictionary."""
    cell_param.filename = filename
    return cell_param


def check_unset_range_mechanisms(cell_param):
    unset_params = []
    for k, v in six.iteritems(cell_param):
        if 'mechanisms' in v:
            for mech_name, mech_params in six.iteritems(v['mechanisms']['range']):
                for param_name, param_value in six.iteritems(mech_params):
                    if param_value is None:
                        unset_params.append('{}.{}.{}'.format(k, mech_name, param_name))
    return unset_params

def set_ephys(cell_param, params=None):
    """Updates cell_param file. 
    
    Parameter names reflect the Hay naming convention.

    Args:
        cell_param (:py:class:`~single_cell_parser.parameters.ParameterSet`): The cell parameter dictionary.
        params (pd.Series): The parameter vector as a pandas Series.

    Raises:
        AssertionError: If the parameter vector is None or not a pandas Series.
        AssertionError: If the provided cell parameters lack a field that is however defined in the parameter vector.
        AssertionError: If some parameters are not set after the update.

    Returns:
        :py:class:`~single_cell_parser.parameters.ParameterSet`: The updated cell_param, with the biphysical parameters set.  
    
    See also:
        See :cite:t:`Hay_Hill_Schuermann_Markram_Segev_2011` for more information.
    """
    assert (
        params is not None
    ), "You must provide the parameter vector here, otherwise all cell parameters will be None."
    assert type(params) == pd.Series, "params must be a pandas Series"
    for k, v in six.iteritems(params):
        scp_param = hay_param_to_scp_neuron_param(k)
        assert hasattr(cell_param, scp_param), "The provided cell parameters have no field called: {}".format(scp_param)
        cell_param[scp_param] = float(v)
    unset_params = check_unset_range_mechanisms(cell_param)
    if len(unset_params) != 0:
        logger.info("The following parameters are not set after set_ephys: {}".format(unset_params))
    return cell_param


def set_param(cell_param, params=None):
    """Updates cell_param given a dict of params in the dot naming convention.
    
    Example::

        cell_param = {'a': {'b': {'c': 2}}}
        params = {'a.b.c': 3}
        set_param(cell_param, params)
        # returns {'a': {'b': {'c': 3}}}
        
    Args:
        cell_param (:py:class`~:py:class:`~single_cell_parser.parameters.ParameterSet`` | dict): The cell parameter nested dictionary.
        params (dict): The parameter flat dictionary.
    
    Returns:
        dict: The updated cell_param.
    """
    for k, v in six.iteritems(params):
        p = cell_param
        for kk in k.split('.')[:-1]:
            p = p[kk]
        p[k.split('.')[-1]] = v
    return cell_param


def set_many_param(cell_param, params=None):
    """Updates cell_param given a dict of params in the dot naming convention.
    
    This method is almost identical to :py:meth:`set_param`, but it has a different behavior when
    a parameter name appears both as a top-level key and as a nested key in :paramref:`params`. In this case, the top-level
    key will be used as the master value.
    
    Example::

        cell_param = {'a': {'b': {'c': 0}}}
        params = {'a': True, 'a.b.c': False}
        set_many_param(cell_param, params)
        # Output: {'a': {'b': {'c': True}}}, NOT {'a': {'b': {'c': False}}}
        
    Args:
        cell_param (:py:class:`~single_cell_parser.parameters.ParameterSet` | dict): The cell parameter nested dictionary.
        params (dict): The parameter flat dictionary.
        
    Returns:
        dict: The updated cell_param.
    """

    master_values = {}

    for k, v in six.iteritems(params):
        if '.' not in k:
            master_values[k] = v

    for k, v in six.iteritems(params):
        if '.' in k:
            stored_value = master_values[k.split('.')[0]]
            p = cell_param
            for key in k.split('.')[1:-1]:
                p = p[key]
            p[k.split('.')[-1]] = stored_value

    return cell_param


def set_hot_zone(cell_param, min_=None, max_=None, outsidescale_sections=None):
    """Insert Ca_LVAst and Ca_HVA channels along the apical dendrite between ``min_`` and ``max_`` distance from the soma.
    
    Args:
        cell_param (:py:class:`:py:class:`~single_cell_parser.parameters.ParameterSet`` | dict): The cell parameter dictionary.
        min_ (float): The minimum distance from the soma.
        max_ (float): The maximum distance from the soma.
        outsidescale_sections (list): A list of sections where the channels should be inserted.
        
    Returns:
        :py:class:`~single_cell_parser.parameters.ParameterSet` | dict: The updated cell_param.
        
    Note:
        This method is specific for a L5PT.
        For more information about the hot zone, refer to :cite:t:`Bast_Guest_Fruengel_Narayanan_de_Kock_Oberlaender_2023`
    """
    hotzone_params = ['ApicalDendrite.Ca_HVA.begin', 'ApicalDendrite.Ca_HVA.end', 
                      'ApicalDendrite.Ca_LVAst.begin', 'ApicalDendrite.Ca_LVAst.end']

    cell_param['ApicalDendrite'].mechanisms.range['Ca_LVAst']['begin'] = min_
    cell_param['ApicalDendrite'].mechanisms.range['Ca_LVAst']['end'] = max_
    cell_param['ApicalDendrite'].mechanisms.range['Ca_HVA']['begin'] = min_
    cell_param['ApicalDendrite'].mechanisms.range['Ca_HVA']['end'] = max_

    if outsidescale_sections is not None:
        assert isinstance(outsidescale_sections, list)
        cell_param['ApicalDendrite'].mechanisms.range['Ca_LVAst'][
            'outsidescale_sections'] = outsidescale_sections
        cell_param['ApicalDendrite'].mechanisms.range['Ca_HVA'][
            'outsidescale_sections'] = outsidescale_sections
        hotzone_params.append('ApicalDendrite.Ca_LVAst.outsidescale_sections')
        hotzone_params.append('ApicalDendrite.Ca_HVA.outsidescale_sections')
        
    unset_params = check_unset_range_mechanisms(cell_param)
    assert all([False if param in unset_params else True for param in hotzone_params]), f"Not all in {hotzone_params} is set"
    logger.info('Following hotzone params are set: {}'.format(hotzone_params))
    return cell_param
