"""Configure modules, functions, methods, classes and attributes so that they are not documented by Sphinx."""

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