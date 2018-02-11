import sys

def get_module_versions():
    out = {}
    for x in sys.modules.keys():
        if not '.' in x:
            try:
                out[x] = sys.modules[x].__version__
            except:
                pass
    return out