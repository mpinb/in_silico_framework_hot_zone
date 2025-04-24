import sys
import importlib
import pkgutil
import config
importlib.reload(config)

# Determine the base IO package dynamically
if config.isf_is_using_mdb() == True:
    base_package = "data_base.model_data_base.IO"
else:
    base_package = "data_base.isf_data_base.IO"

# Import the base package
selected_IO = importlib.import_module(base_package)

# Recursively register all subpackages and modules
for finder, name, ispkg in pkgutil.walk_packages(selected_IO.__path__, prefix=selected_IO.__name__ + "."):
    module = importlib.import_module(name)
    # Add the module to the current namespace
    sys.modules[__name__ + name[len(base_package):]] = module

# Replace the current module with the selected IO module
sys.modules[__name__] = selected_IO