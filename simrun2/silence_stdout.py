import sys
import os

def silence_stdout(fun):
    '''robustly silences a function and restores stdout afterwars'''
    def silent_fun(*args, **kwargs):
        stdout_bak = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            res = fun(*args, **kwargs)
        except:
            raise
        finally:
            sys.stdout = stdout_bak
        return res
    return silent_fun
