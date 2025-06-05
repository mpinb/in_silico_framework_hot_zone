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
"""Create a folder and return it as a ManagedFolder object."""

import os
# import cloudpickle
import compatibility
from . import parent_classes
import json
from data_base.utils import calc_recursive_filetree, bcolors, colorize_str
from pathlib import Path

def check(obj):
    """Check whether the object can be saved with this dumper
    
    Args:
        obj (object): Object to be saved
        
    Returns:
        bool: Whether the object is None. This dumper requires no object to be saved.
    """
    return obj is None  #isinstance(obj, np) #basically everything can be saved with pickle


## this was used to store more data in the ManagedFolder
## it turned out to be complex to inherit from an immutable datatype
## https://stackoverflow.com/questions/2673651/inheritance-from-str-or-int
##
# class ManagedFolder(str):
#     def __new__(cls, s, db):
#         obj = str.__new__(cls, s)
#         obj.db = db
#         return obj
#     def __init__(self, s, db):
#         str.__init__(s)
#         self.db = db
#     def join(self, *args):
#         return ManagedFolder(os.path.join(self, *args), self.db)
#     def __reduce__(self):
#         return self.__class__, (str(self), self.db)


class ManagedFolder(str):
    """Wrapper class for a folder path    
    """
    def join(self, *args):
        """Get a subfolder of the current folder
        
        Args:
            *args: Subfolder names
            
        Returns:
            :py:class:`~data_base.isf_data_base.IO.LoaderDumper.just_create_folder.ManagedFolder`: Subfolder
        """
        return ManagedFolder(os.path.join(self, *args))

    def listdir(self):
        """List the files in the folder"""
        return [f for f in os.listdir(self) if not f in ['Loader.pickle', 'Loader.json', 'metadata.json']]
    
    def keys(self):
        return self.listdir()

    def get_file(self, suffix):
        '''Get the files that end with the specified suffix.
        
        Args:
            suffix (str): Suffix of the file
            
        Raises:
            ValueError: If there are no files with the specified suffix or more than one file with the specified suffix
        
        Returns:
            str: The filepath of the file with :paramref:`suffix` (only if there is exactly one file with this suffix)
        '''
        l = [f for f in os.listdir(self) if f.endswith(suffix)]
        if len(l) == 0:
            raise ValueError(
                'The folder {} does not contain a file with the suffix {}'.
                format(self, suffix))
        elif len(l) > 1:
            raise ValueError(
                'The folder {} contains several files with the suffix {}'.
                format(self, suffix))
        else:
            return os.path.join(self, l[0])
        
    def __getitem__(self, key):
        if not isinstance(key, str):
            raise ValueError(f"Key must be string but is {type(key)}")
        if key in ['.', '..']:
            raise NotImplementedError()
        path = self.join(key)
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")
        if not os.path.isdir(path):
            raise ValueError(f"{path} is not a directory")
        return path
    
    def __delitem__(self):
        shutil.rmtree(self)
        
    def __repr__(self):
        """Get a string representation of the database.
        
        This method is called when you print the database object.
        
        Example::
        
            >>> print(my_data_base)
            <data_base.isf_data_base.ISFDataBase object at 0x7f0d8d3d4a90>
            Located at <path>
            db
            └── key
                ├── subkey1
                ├── subkey2
                ... (n more)
        
        Returns:
            str: A string representation of the database.
        """
        return self._get_str()  # print with default depth and max_lines
    
    def _get_str(self, depth=0, max_depth=2, max_lines=20, all_files=False, max_lines_per_key=3, color=True):
        """Fetches a string representation for this db in a tree structure.
        
        This is internal API and should never be called directly.
        
        Args:
            max_depth (int, optional): How deep you want the filestructure to be. Defaults to 2.
            max_lines (int, optional): How long you want your total filelist to be. Defaults to 20.
            max_lines_per_key (int, optional): How many lines to print per key. Useful to limit visual output of subdatabases. Defaults to 3.
            all_files (bool, optional): Whether to print all files (including e.g. ``Loader.json``), or only keys. Defaults to False.

        Returns:
            str: A string representation of this db in a tree structure.
        """

        str_ = ['<{}.{} object at {}>'.format(self.__class__.__module__, self.__class__.__name__, hex(id(self)))]
        str_.append("Located at {}".format(self))
        # str_.append("{1}DataBases{0} | {2}Directories{0} | {3}Keys{0}".format(
        #     bcolors.ENDC, bcolors.OKGREEN, bcolors.WARNING, bcolors.OKCYAN) )
        if color:
            str_.append(colorize_str(self, bcolors.OKGREEN))
        else:
            str_.append(self)
        lines = calc_recursive_filetree(
            self, Path(self), 
            depth=depth, max_depth=max_depth, 
            max_lines_per_key=max_lines_per_key, 
            all_files=all_files, 
            max_lines=max_lines,
            colorize=color)
        for line in lines:
            str_.append(line)
        return "\n".join(str_)
    


class Loader(parent_classes.Loader):
    """Load a :py:class:`~data_base.isf_data_base.IO.LoaderDumper.just_create_folder.ManagedFolder` object from a folder path
    """
    def get(self, savedir):
        """Get a :py:class:`~data_base.isf_data_base.IO.LoaderDumper.just_create_folder.ManagedFolder` object from a folder path
        
        Args:
            savedir (str): Folder path
        """
        #return savedir
        return ManagedFolder(savedir)


def dump(obj, savedir):
    """Create a folder
    
    Args:
        obj (None, optional): 
            This dumper requires no object to be saved. Only here for consistent API with other dumpers.
            If an object is specified, it is ignored.
        savedir (str): Folder path
    """
    with open(os.path.join(savedir, 'Loader.json'), 'w') as f:
        json.dump({'Loader': __name__}, f)

