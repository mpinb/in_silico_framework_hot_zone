"""Configure modules, functions, methods, classes and attributes so that they are not documented by Sphinx."""
import pkgutil, importlib, os
from unittest.mock import patch


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

def safe_import_module(module_name):
    """Import a module safely, ignoring sys.exit and KeyError exceptions."""
    try:
        with patch('sys.exit', lambda x: None):
            module = importlib.import_module(module_name)
        return module
    except ImportError:
        print(f"Failed to import module {module_name}")
    except KeyError:
        print(f"KeyError encountered while importing module {module_name}")
    return None
    

def find_modules_with_tag(source_dir, tag=":skip-doc:"):
    """Recursively find all modules with a specific tag in their docstring."""
    modules_with_tag = []

    def check_module(module_name):
        """Check if a module or any of its submodules has the specific tag in its docstring."""
        module = safe_import_module(module_name)
        if module is None:
            return False
        if module.__doc__ and tag in module.__doc__:
            if module.__name__.endswith('.__init__'):
                module_name = module.__name__[:-9]
            modules_with_tag.append(module_name)
            return True
        if hasattr(module, '__path__'):  # Check if the module is a package
            for _, submodule_name, is_pkg in pkgutil.iter_modules(module.__path__):
                full_submodule_name = f"{module_name}.{submodule_name}"
                if check_module(full_submodule_name):
                    modules_with_tag.append(full_submodule_name)
                    return True
        return False

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".py"):
                module_path = os.path.relpath(os.path.join(root, file), source_dir)
                module_name = module_path.replace(os.sep, ".")[:-3]  # Remove .py extension
                check_module(module_name)

    return modules_with_tag