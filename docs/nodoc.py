"""Configure modules, functions, methods, classes and attributes so that they are not documented by Sphinx."""

from sphinx.ext.autosummary import Autosummary
import importlib

def skip_member(app, what, name, obj, skip, options):
    """Skip members if they have the :skip-doc: tag in their docstring."""
    # Debug print to check what is being processed
    # print(f"Processing {what}: {name}")
    
    # skip special members, except __get__ and __set__
    if name.startswith('__') and name.endswith('__') and name not in ['__get__', '__set__']:
        return True
    
    # Skip if it has the :skip-doc: tag
    if obj.__doc__ and ':skip-doc:' in obj.__doc__:
        if ':skip-doc:' in obj.__doc__:
            # print(f"Docstring for {name}: {obj.__doc__}")
            print(f"Skipping {what}: {name} due to :skip-doc: tag")
            return True
    
    # Skip inherited members
    if hasattr(obj, '__objclass__') and obj.__objclass__ is not obj.__class__:
        return True
    
    return skip

class CustomAutosummary(Autosummary):
    """Skip modules
    
    We use a custom tag :skip-doc: to skip the documentation of members using the autodoc-skip-member hook.
    This hook seems to not work well with the autosummary extension: members that are skipped by autodoc-skip-member still show up in the set of autosummary templates for example.
    This seems to be fixed in https://github.com/sphinx-doc/sphinx/issues/6798
    I dont know if we use this version of sphinx, but it for sure does not work in this project when relying on only the autodoc-skip-member hook.
    For this reason, we overload the Autosummary get_items method here to filter out members that have the :skip-doc: tag in their docstring.
    """
    def get_items(self, names):
        items = super().get_items(names)
        filtered_items = []
        for name, sig, summary, real_name in items:
            module_name = real_name.split('.')[0]
            module = importlib.import_module(module_name)
            if module.__doc__ and ':skip-doc:' in module.__doc__:
                continue
            filtered_items.append((name, sig, summary, real_name))
        return filtered_items
