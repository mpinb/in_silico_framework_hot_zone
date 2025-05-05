"""Handle :ref:`params_file_format` files in ISF.
"""

from collections.abc import MutableMapping
import json, re, neuron
from data_base.dbopen import dbopen, resolve_modular_db_path

def _read_params_to_dict(filename):
    filename = resolve_modular_db_path(filename)
    with dbopen(filename, "r") as f:
        content = f.read()

    # Replace single quotes with double quotes
    content = content.replace("'", '"')

    # Remove trailing commas using regex
    content = re.sub(r",(\s*[}\]])", r"\1", content)

    # Replace Python-style tuples (x, y) with JSON arrays [x, y]
    content = re.sub(r"\(([^()]+)\)", r"[\1]", content)
    
    try:
        params_dict = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding .param file with JSON parsing: {e}")
    return params_dict


def build_parameters(filename):
    """Read in a :ref:`param_file_format` file and return a ParameterSet object.

    Args:
        filename (str): path to the parameter file

    Returns:
        :py:class:`~single_cell_parser.parameters.ParameterSet`: The parameter file as a :py:class:`~single_cell_parser.parameters.ParameterSet` object.
    """
    data = _read_params_to_dict(filename)
    return ParameterSet(data)


def load_NMODL_parameters(parameters):
    """Load NMODL mechanisms from paths in parameter file.

    Parameters are added to the NEURON namespace by executing string Hoc commands.

    See also: https://www.neuron.yale.edu/neuron/static/new_doc/programming/neuronpython.html#important-names-and-sub-packages

    Args:
        parameters (:py:class:`~single_cell_parser.parameters.ParameterSet` | dict):
            The neuron parameters to load.
            Must contain the key `NMODL_mechanisms`.
            May contain the key `mech_globals`.

    Returns:
        None. Adds parameters to the NEURON namespace.
    """
    for mech in list(parameters.NMODL_mechanisms.values()):
        neuron.load_mechanisms(mech)
    try:
        for mech in list(parameters.mech_globals.keys()):
            for param in parameters.mech_globals[mech]:
                paramStr = param + "_" + mech + "="
                paramStr += str(parameters.mech_globals[mech][param])
                print("Setting global parameter", paramStr)
                neuron.h(paramStr)
    except AttributeError:
        pass


class ParameterSet(MutableMapping):
    """
    A wrapper class for dictionaries that allows attribute-style access to keys.
    Works recursively for nested dictionaries.
    """

    def __init__(self, data):
        """
        Args:
            data (dict): The dictionary to wrap.

        Raises:
            TypeError: If the input data is not a dictionary.
        """
        if isinstance(data, str):
            data = _read_params_to_dict(data)
        if not (isinstance(data, dict) or isinstance(data, ParameterSet)):
            raise TypeError(f"ParameterSet can only wrap dictionaries. You provided {type(data)}.")
        self._data = {key: self._wrap(value) for key, value in data.items()}

    def _wrap(self, value):
        """
        Recursively wrap dictionaries as ParameterSet objects.
        """
        if isinstance(value, dict):
            return ParameterSet(value)
        elif isinstance(value, list):
            return [self._wrap(item) for item in value]
        return value

    def __getattr__(self, name):
        """
        Allow attribute-style access to dictionary keys.
        """
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'ParameterSet' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """
        Allow setting attributes, except for internal attributes.
        """
        if name == "_data":
            super().__setattr__(name, value)
        else:
            self._data[name] = self._wrap(value)

    def __delattr__(self, name):
        """
        Allow deleting attributes.
        """
        if name in self._data:
            del self._data[name]
        else:
            raise AttributeError(f"'ParameterSet' object has no attribute '{name}'")

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = self._wrap(value)

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def to_dict(self):
        """
        Convert the ParameterSet back to a regular dictionary.
        """
        def unwrap(value):
            if isinstance(value, ParameterSet):
                return value.to_dict()
            elif isinstance(value, list):
                return [unwrap(item) for item in value]
            return value

        return {key: unwrap(value) for key, value in self._data.items()}

    def __repr__(self):
        """
        String representation of the ParameterSet object.
        """
        return f"ParameterSet({self._data})"

    def __getstate__(self):
        """
        Get the state of the ParameterSet for pickling.
        """
        return self.to_dict()

    def __setstate__(self, state):
        """
        Set the state of the ParameterSet after unpickling.
        """
        self._data = {key: self._wrap(value) for key, value in state.items()}