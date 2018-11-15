'''
create plots corresponding to Lang et al. Fig. 5
'''

import sys
import os.path
import glob
import numpy as np
import matplotlib.pyplot as plt

def create_summary(dirName, cellTypeNames):
    fnames = []
    scan_directory(dirName, fnames, cellTypeNames)
    print 'Analyzing %d files!' % len(fnames)
    allData = {}
    for fname in fnames:
        splitName = fname.split('_')
        gAMPA = splitName[-8]
        gNMDA = splitName[-5]
        if gAMPA not in allData.keys():
            allData[gAMPA] = {}
        if allData[gAMPA].has_key(gNMDA):
            fileData = create_plots(fname)
            allData[gAMPA][gNMDA]['Vm'].extend(fileData[0])
            allData[gAMPA][gNMDA]['T'].extend(fileData[1])
        else:
            allData[gAMPA][gNMDA] = {}
            fileData = create_plots(fname)
            allData[gAMPA][gNMDA]['Vm'] = list(fileData[0][:])
            allData[gAMPA][gNMDA]['T'] = list(fileData[1][:])
    
    tPSPStart = 100.0
#    detectionThreshold = 0.1 # VPM
    detectionThreshold = 0.15 # intracortical
    summaryData = {}
    for gAMPAStr in allData.keys():
        summaryData[gAMPAStr] = {}
        for gNMDAStr in allData[gAMPAStr].keys():
            Vm = allData[gAMPAStr][gNMDAStr]['Vm']
            tPSP = allData[gAMPAStr][gNMDAStr]['T']
            VmDetected = []
            tPSPDetected = []
            for i in range(len(Vm)):
                if Vm[i] > detectionThreshold:
                    VmDetected.append(Vm[i])
                    tPSPDetected.append(tPSP[i])
            epspMean = np.mean(VmDetected)
            epspStd = np.std(VmDetected)
            epspMed = np.median(VmDetected)
            epspMin = np.min(VmDetected)
            epspMax = np.max(VmDetected)
            tMean = np.mean(np.array(tPSPDetected)-tPSPStart)
            tStd = np.std(np.array(tPSPDetected)-tPSPStart)
            tMed = np.median(np.array(tPSPDetected)-tPSPStart)
            summaryData[gAMPAStr][gNMDAStr] = epspMean, epspStd, epspMed, epspMin, epspMax, tMean, tStd, tMed
    
    gAMPAStr = summaryData.keys()
    gAMPAStr.sort()
    #gNMDAStr = summaryData[gAMPAStr[0]].keys()
    #gNMDAStr.sort()
    outName = dirName
    if not dirName.endswith('/'):
        outName += '/'
    for cellTypeName in cellTypeNames:
        outName += cellTypeName + '_'
    outName += 'summary.csv'
    with open(outName, 'w') as outFile:
        header = '# gAMPA\tgNMDA\tepspMean\tepspStd\tepspMed\tepspMin\tepspMax\ttMean\ttStd\ttMed\n'
        outFile.write(header)
        for i in range(len(gAMPAStr)):
            gAMPA = gAMPAStr[i]
            gNMDA = summaryData[gAMPA].keys()[0]
            line = gAMPA
            line += '\t'
            line += gNMDA
            for i in range(8):
                line += '\t'
                line += str(summaryData[gAMPA][gNMDA][i])
            line += '\n'
            outFile.write(line)
#    for fname in fnames:
#        splitName = fname.split('_')
#        gEx = splitName[2]
#        summaryData[gEx] = create_plots(fname)
#    
#    gExStr = summaryData.keys()
#    gExStr.sort()
#    outName = dirName
#    if not dirName.endswith('/'):
#        outName += '/'
#    outName += 'summary.csv'
#    with open(outName, 'w') as outFile:
#        header = '# gAMPA\tgNMDA\tepspMean\tepspStd\tepspMed\tepspMin\tepspMax\ttMean\ttStd\ttMed\n'
#        outFile.write(header)
#        for gEx in gExStr:
#            line = gEx
#            line += '\t'
#            line += gEx
#            for i in range(8):
#                line += '\t'
#                line += str(summaryData[gEx][i])
#            line += '\n'
#            outFile.write(line)

def scan_directory(path, fnames, cellTypeNames):
    for fname in glob.glob(os.path.join(path, '*')):
        if os.path.isdir(fname):
            scan_directory(fname, fnames, cellTypeNames)
        elif fname.endswith('vmax.csv'):
            for cellTypeName in cellTypeNames:
                if cellTypeName in fname and fname not in fnames:
                    fnames.append(fname)
        else:
            continue
            
def create_plots(fname):
    synID, somaV, somaT = np.loadtxt(fname, skiprows=1, unpack=True)
    
#    outName = fname[:-4]
#    epspName = outName + '_epsp_hist.pdf'
#    dtName = outName + '_dt_hist.pdf'
#    
#    epspMean = np.mean(somaV)
#    epspStd = np.std(somaV)
#    epspMed = np.median(somaV)
#    plt.figure()
#    plt.hist(somaV,bins=9)
#    plt.xlabel('uEPSP amplitude at soma [$mV$]')
#    plt.ylabel('frequency')
#    titleStr = 'mean uEPSP = %.2f$\pm$%.2f $mV$; median = %.2f' % (epspMean,epspStd,epspMed)
#    plt.title(titleStr)
#    plt.savefig(epspName, bbox_inches=0)
#    
#    tMean = np.mean(somaT-10.0)
#    tStd = np.std(somaT-10.0)
#    tMed = np.median(somaT-10.0)
#    plt.figure()
#    plt.hist(somaT-10.0,bins=12)
#    plt.xlabel('time to uEPSP peak at soma [$ms$]')
#    plt.ylabel('frequency')
#    titleStr2 = 'mean dt = %.2f$\pm$%.2f $mV$; median = %.2f' % (tMean,tStd,tMed)
#    plt.title(titleStr2)
#    plt.savefig(dtName, bbox_inches=0)
    
    try:
        return list(somaV), list(somaT)
    except TypeError:
        return [somaV], [somaT]

if __name__ == '__main__':
    dirName = sys.argv[1]
#    create_plots(fname)
    cellTypeNames = sys.argv[2:]
    create_summary(dirName, cellTypeNames)
    
    