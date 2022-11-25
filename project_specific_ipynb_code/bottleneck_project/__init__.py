from tqdm import tqdm
import random, string
import model_data_base.mdbopen
import torch
from sklearn.metrics import roc_auc_score

# idk why these modules are necessary but i'm too scared to delete them