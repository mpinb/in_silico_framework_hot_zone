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
import os
# import cloudpickle
import compatibility
from . import parent_classes


def check(obj):
    '''checks wherther obj can be saved with this dumper'''
    return obj is None  #isinstance(obj, np) #basically everything can be saved with pickle


## this was used to store more data in the ManagedFolder
## it turned out to be complex to inherit from an immutable datatype
## https://stackoverflow.com/questions/2673651/inheritance-from-str-or-int
##
# class ManagedFolder(str):
#     def __new__(cls, s, mdb):
#         obj = str.__new__(cls, s)
#         obj.mdb = mdb
#         return obj
#     def __init__(self, s, mdb):
#         str.__init__(s)
#         self.mdb = mdb
#     def join(self, *args):
#         return ManagedFolder(os.path.join(self, *args), self.mdb)
#     def __reduce__(self):
#         return self.__class__, (str(self), self.mdb)


class ManagedFolder(str):

    def join(self, *args):
        return ManagedFolder(os.path.join(self, *args))

    def listdir(self):
        return [f for f in os.listdir(self) if not f == 'Loader.pickle']

    def get_file(self, suffix):
        '''if folder only contains one file of specified suffix, this file is returned'''
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


class Loader(parent_classes.Loader):

    def get(self, savedir):
        #return savedir
        return ManagedFolder(savedir)


def dump(obj, savedir):
    #     with open(os.path.join(savedir, 'Loader.pickle'), 'wb') as file_:
    #         cloudpickle.dump(Loader(), file_)
    compatibility.cloudpickle_fun(Loader(),
                                  os.path.join(savedir, 'Loader.pickle'))
