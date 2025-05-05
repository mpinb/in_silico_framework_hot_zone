"""Handle :ref:`params_file_format` files in ISF.
"""

from collections.abc import MutableMapping

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