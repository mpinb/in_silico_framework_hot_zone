import pandas as pd

def get_EPSP_measurement():
    EPSP_mean = [0.49,0.49,0.35,0.47,0.46,0.44,0.44,0.571]
    EPSP_med = [0.35,0.35,0.33,0.33,0.36,0.31,0.31,0.463]
    EPSP_max = [1.9,1.9,1.0,1.25,1.5,1.8,1.8,1.18]
    celltypes = ['L2', 'L34', 'L4', 'L5st', 'L5tt', 'L6cc', 'L6ct', 'VPM_C2']
    return pd.DataFrame({'EPSP_mean_measured': EPSP_mean, 
                           'EPSP_med_measured':EPSP_med, 
                           'EPSP_max_measured': EPSP_max}, index = celltypes)
    
color_cellTypeColorMap = {'L1': 'cyan', 'L2': 'dodgerblue', 'L34': 'blue', 'L4py': 'palegreen',\
                    'L4sp': 'green', 'L4ss': 'lime', 'L5st': 'yellow', 'L5tt': 'orange',\
                    'L6cc': 'indigo', 'L6ccinv': 'violet', 'L6ct': 'magenta', 'VPM': 'black',\
                    'INH': 'grey', 'EXC': 'red', 'all': 'black', 'PSTH': 'blue'}

excitatory = ['L6cc', 'L2', 'VPM', 'L4py', 'L4ss', 'L4sp', 'L5st', 'L6ct', 'L34', 'L6ccinv', 'L5tt', 'Generic']
inhibitory = ['SymLocal1', 'SymLocal2', 'SymLocal3', 'SymLocal4', 'SymLocal5', 'SymLocal6', 'L45Sym', 'L1', 'L45Peak', 'L56Trans', 'L23Trans', 'GenericINH', 'INH']
