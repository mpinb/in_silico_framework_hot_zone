
# coding: utf-8

# In[1]:


import os
import sys

nb_dir = os.path.split(os.getcwd())[0]
if (nb_dir not in sys.path):
    sys.path.append(nb_dir)


# In[2]:


amDataPath = str('../data/neuron1/am/')
tifDataPath = str('../data/neuron1/tif/max_z_projections/')
amOutputPath = str('../output/neuron1/am/')
outputFolderPath = str('../output/neuron1/am/')
tifOutputPath = str('../output/neuron1/tif/')


# In[3]:


import radii as radi
getRad = radi.exRadSets.exRadSets


# In[4]:


getRad(amDataPath, tifDataPath, outputFolderPath)

