
# coding: utf-8

# In[1]:


import os
import sys

nb_dir = os.path.split(os.getcwd())[0]
if (nb_dir not in sys.path):
    sys.path.append(nb_dir)


# In[2]:


amDataPath = str('../data/report/input/am/')
tifDataPath = str('../data/report/input/tif/max_z_projections/')
amOutputPath = str('../data/report/output/am/')
outputFolderPath = str('../data/report/output/am/')
tifOutputPath = str('../data/report/output/tif/')


# In[3]:


import radii as radi
getRad = radi.exRadSets.exRadSets


# In[4]:


getRad(amDataPath, tifDataPath, outputFolderPath)

