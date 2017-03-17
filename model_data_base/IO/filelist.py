'''
Created on Aug 15, 2016

@author: arco
'''

import os, glob
import fnmatch

def scan_directory(path, fnames, suffix):
    for fname in glob.glob(os.path.join(path, '*')):
        if os.path.isdir(fname):
            scan_directory(fname, fnames, suffix)
        elif fname.endswith(suffix):
            fnames.append(fname)
        else:
            continue
        
def scan_directory_r(path, suffix):
    out = []
    scan_directory(path, out, suffix)
    return out

def make_file_list(directory, filename):
#     prefix = directory
#     file_list = scan_directory_r(directory, filename)
#     file_list = [os.path.relpath(full_path, prefix) for full_path in file_list]
#     return pd.Series(file_list)
    matches = []
    for root, dirnames, filenames in os.walk('src'):
        for filename in fnmatch.filter(filenames, '*'+filename):
            matches.append(os.path.join(root, filename))
            
    return matches