'''

'''

import numpy as np
from data_base.dbopen import dbopen

__author__ = 'Robert Egger'
__date__ = '2012-11-03'


def vm_mean(vVec, tVec, tStim, dtStim):
    '''Computes the mean voltage during a stimulation time window

    Args:
        vVec (list): voltage vector
        tVec (list): time vector
        tStim (float): start time of stimulation
        dtStim (float): duration of stimulation

    Returns:
        float: mean voltage during stimulation time window
    '''
    t = np.array(tVec)
    v = np.array(vVec)
    tEnd = tStim + dtStim
    iBegin, iEnd = 0, 0
    for i in range(1, len(t)):
        if t[i - 1] < tStim and t[i] >= tStim:
            iBegin = i
        if t[i - 1] < tEnd and t[i] >= tEnd:
            iEnd = i
    v_ = v[iBegin:iEnd]
    vss = np.mean(v_)
    return vss


def vm_std(vVec, tVec, tStim, dtStim):
    '''Computes the standard deviation of the voltage during a stimulation time window

    Args:
        vVec (list): voltage vector
        tVec (list): time vector
        tStim (float): start time of stimulation
        dtStim (float): duration of stimulation

    Returns:
        float: standard deviation of the voltage during stimulation time window
    '''
    t = np.array(tVec)
    v = np.array(vVec)
    tEnd = tStim + dtStim
    iBegin, iEnd = 0, 0
    for i in range(1, len(t)):
        if t[i - 1] < tStim and t[i] >= tStim:
            iBegin = i
        if t[i - 1] < tEnd and t[i] >= tEnd:
            iEnd = i
    v_ = v[iBegin:iEnd]
    vss = np.std(v_)
    return vss


def compute_mean_psp_amplitude(vTraces, tStim, dt, width=35.0, t_delay=15.0):
    """Compute the mean amplitude of all PSPs across multiple voltage traces.
    
    The post-synaptic potential (PSP) amplitude is the maximum membrane voltage deflection
    between :paramref:`tStim` + ``delay`` and :paramref:`tStim` + ``delay`` + ``width``,
    where ``delay`` and ``width`` are :math:`15.0` and :math:`35.0 \, ms` respectively.

    Args:
        vTraces (list): List of voltage traces
        tStim (float): Time of stimulation
        dt (float): Time step

    Returns:
        tuple: A tuple containing the delay and the mean amplitude of the PSPs.    
    """
    amplitudes = []
    for trace in vTraces:
        # +0.5 is simply for rounding up
        begin = int((tStim + t_delay) / dt + 0.5)
        end = int((tStim + t_delay + width) / dt + 0.5)
        amplitudes.append(np.max(trace[begin:end]))
    return [t_delay], [np.mean(amplitudes)]


def compute_vm_std_windows(vStd, tStim, dt, width=35.0, window_start_times=None):
    if window_start_times is None:
        window_start_times = [-50.0, 15.0]
    """Compute the standard deviation of the voltage during different time windows.
    
    Time windows are defined as :math:`[-50.0, 15.0] \, ms` and :math:`[15.0, 50.0] \, ms`.
    
    Args:
        vStd (list): List of standard deviations of the voltage
        tStim (float): Time of stimulation
        dt (float): Time step
        
    Returns:
        tuple: A tuple containing the time windows and the average standard deviations.
    """
    avgStds = []
    for t in window_start_times:
        begin = int((tStim + t) / dt + 0.5)
        end = int((tStim + t + width) / dt + 0.5)
        avgStds.append(np.mean(vStd[begin:end]))
    return np.array(window_start_times), np.array(avgStds)


def compute_vm_histogram(vTraces, bins=None):
    if bins is None:
        bins = np.array([-80.0 + i * 2.0 for i in range(21)])
    vPoints = vTraces.flatten()
    hist, bin_edges = np.histogram(vPoints, bins=bins)
    norm = np.max(hist)
    hist = 1.0 / norm * hist
    return hist, bins


def compute_uPSP_amplitude(t, v, tSyn, isEPSP=True, t_width_baseline=10.0):
    '''
    simple method for determining amplitude of
    unitary PSP starting at time tSyn:
    Determine baseline as median Vm during 10ms
    preceding the PSP
    In case of IPSPs set isEPSP=False
    '''
    if len(t) != len(v):
        errstr = 'Time vector and membrane potential vector do not match'
        raise RuntimeError(errstr)
    
    # find baseline
    beginBin = 0
    endBin = 0
    while t[endBin] < tSyn:
        endBin += 1
    while t[beginBin] < tSyn - t_width_baseline:
        beginBin += 1
    endBin -= 1
    beginBin -= 1
    if beginBin < 0:
        beginBin = 0
    baseline = np.median(v[beginBin:endBin])

    # find amplitude wrt baseline
    if isEPSP:
        return np.max(v[endBin:]) - baseline
    else:
        return np.min(v[endBin:]) - baseline


def simple_spike_detection(
        t,
        v,
        tBegin=None,
        tEnd=None,
        threshold=0.0,
        mode='regular'):
    '''
    Simple spike detection method. Identify
    spike times within optional window [tBegin, tEnd]
    by determining threshold crossing times from below
    supported modes:
    regular: absolute threshold crossing
    differential: threshold crossing of dv/dt
    '''
    if len(t) != len(v):
        errstr = 'Dimensions of time vector and membrane potential vector not matching'
        raise RuntimeError(errstr)

    tSpike = []
    beginIndex = 1
    endIndex = len(t)
    if tBegin is not None:
        for i in range(1, len(t)):
            if t[i - 1] < tBegin and t[i] >= tBegin:
                beginIndex = i
                break
    if tEnd is not None:
        for i in range(1, len(t)):
            if t[i - 1] < tEnd and t[i] >= tEnd:
                endIndex = i
                break

    if mode == 'regular':
        for i in range(beginIndex, endIndex):
            if v[i - 1] < threshold and v[i] >= threshold:
                tSpike.append(t[i])

    if mode == 'slope':
        dvdt = np.diff(v) / np.diff(t)
        for i in range(beginIndex, endIndex):
            if dvdt[i - 1] < threshold and dvdt[i] >= threshold:
                tSpike.append(t[i])

    return tSpike


def PSTH_from_spike_times(
        spikeTimeVectors,
        binSize=1.0,
        tBegin=None,
        tEnd=None,
        aligned=True):
    '''
    Calculate PSTH from vectors of spike times
    with optional bin size and time window [tBegin, tEnd].
    If aligned=True, then the bins are aligned to
    integer multiples of the bin size.
    '''
    norm = len(spikeTimeVectors)
    allSpikeTimes = []
    for spikeTimeVec in spikeTimeVectors:
        for spikeTime in spikeTimeVec:
            allSpikeTimes.append(spikeTime)

    if tBegin is None:
        tBegin = np.min(allSpikeTimes)
    if tEnd is None:
        tEnd = np.max(allSpikeTimes)
    if aligned:
        if tBegin >= 0:
            tBegin = int(tBegin / binSize) * binSize
        else:
            tBegin = int(tBegin / binSize - 1.0) * binSize
        tEnd = (int(tEnd / binSize) + 1.0) * binSize


    # print 'binSize = %.2f' % binSize
    # print 'tBegin = %.2f' % tBegin
    # print 'tEnd = %.2f' % tEnd

    bins = np.arange(tBegin, tEnd, binSize)
    hist, bins = np.histogram(allSpikeTimes, bins)
    if norm:
        hist = 1.0 / norm * hist

    return hist, bins


class RecordingSiteManager(object):
    '''
    simple class providing methods to handle recording
    sites defined by Amira LandmarkSet
    '''
    recordingSites = None
    cell = None

    def __init__(self, landmarkFilename, cell):
        landmarks = self._read_landmark_file(landmarkFilename)
        self.cell = cell
        self.recordingSites = []
        for i in range(len(landmarks)):
            landmark = np.array(landmarks[i])
            newRecSite = self.set_up_recording_site(
                landmark, 
                i,
                landmarkFilename)
            self.recordingSites.append(newRecSite)

    def set_up_recording_site(self, location, ID, filename):
        '''
        determine section/segment on cell corresponding
        to recording site location and create new
        RecordingSite
        '''
        # inaccurate version
        #        minDist = 1e9
        #        minSecID = None
        #        minSegID = None
        #        for i in range(len(self.cell.sections)):
        #            sec = self.cell.sections[i]
        #            for j in range(len(sec.segPts)):
        #                pt = sec.segPts[j]
        #                dist = np.sqrt(np.dot(pt-location, pt-location))
        #                if(dist < minDist):
        #                    minDist = dist
        #                    minSecID = i
        #                    minSegID = j
        #
        #        sec = self.cell.sections[minSecID]
        #        somaDist = self.cell.distance_to_soma(sec, sec.relPts[minSegID])
        #        splitName = filename.split('/')[-1]
        #        tmpIndex = splitName.find('.landmarkAscii')
        #        label = splitName[:tmpIndex] + '_ID_%03d_sec_%03d_seg_%03d_somaDist_%.1f' % (ID, minSecID, minSegID, somaDist)
        #        newRecSite = RecordingSite(minSecID, minSegID, label)
        #        return newRecSite

        # precise version
        minDist = 1e9
        minSecID = None
        minSegID = None
        minx = None
        minSegx = None
        for i in range(len(self.cell.sections)):
            sec = self.cell.sections[i]
            for j in range(len(sec.pts)):
                pt = sec.pts[j]
                ptx = sec.relPts[j]
                dist = np.sqrt(np.dot(pt - location, pt - location))
                if (dist < minDist):
                    minDist = dist
                    minSecID = i
                    minx = ptx
        mindx = 1.0e9
        for i in range(len(self.cell.sections[minSecID].segx)):
            x = self.cell.sections[minSecID].segx[i]
            dx = abs(x - minx)
            if dx < mindx:
                mindx = dx
                minSegx = x
                minSegID = i

        sec = self.cell.sections[minSecID]
        somaDist = self.cell.distance_to_soma(sec, minSegx)
        splitName = filename.split('/')[-1]
        tmpIndex = splitName.find('.landmarkAscii')
        label = splitName[:
                          tmpIndex] + '_ID_%03d_sec_%03d_seg_%03d_x_%.3f_somaDist_%.1f' % (
                              ID, minSecID, minSegID, minSegx, somaDist)
        newRecSite = RecordingSite(minSecID, minSegID, label)
        return newRecSite

    def update_recordings(self):
        '''
        copy recorded voltages at recording sites
        '''
        for recordingSite in self.recordingSites:
            secID = recordingSite.secID
            segID = recordingSite.segID
            vTrace = np.array(self.cell.sections[secID].recVList[segID])
            recordingSite.vRecordings.append(vTrace)

    def _read_landmark_file(self, landmarkFilename):
        '''
        returns list of (x,y,z) points
        '''
        if not landmarkFilename.endswith('.landmarkAscii'):
            errstr = 'Wrong input format: has to be landmarkAscii format. Path: {p}'.format(
                p=landmarkFilename)
            raise RuntimeError(errstr)

        landmarks = []
        with dbopen(landmarkFilename, 'r') as landmarkFile:
            readPoints = False
            for line in landmarkFile:
                stripLine = line.strip()
                if not stripLine:
                    continue
                if stripLine[:2] == '@1':
                    readPoints = True
                    continue
                if readPoints:
                    splitLine = stripLine.split()
                    x = float(splitLine[0])
                    y = float(splitLine[1])
                    z = float(splitLine[2])
                    landmarks.append((x, y, z))

        return landmarks


class RecordingSite(object):
    '''
    light-weight container holding information
    about recording site:
    section and segment ID of attached cell,
    identifier label, and list of recorded voltage vectors
    '''
    secID = None
    segID = None
    label = None
    vRecordings = None

    def __init__(self, secID, segID, label):
        self.secID = secID
        self.segID = segID
        self.label = label
        self.vRecordings = []


class SpikeInit:
    '''
    Provides methods for analysis of spike initiation
    in in vivo/ in vitro simulations. Can be used to
    obtain features of spike shape, adaptation etc...
    '''

    def __init__(self):
        #        TODO: implement
        pass

    @staticmethod
    def vm_steady_state(cell, tVec, tStim, dtStim):
        '''
        computes "steady-state" subthreshold voltage from
        voltage trace during stimulation time window
        [tStim, tStim+dtStim]
        this can be used to define voltage spike threshold
        at current intensities just below AP initiation
        '''
        t = np.array(tVec)
        v = np.array(cell.soma.recVList[0])
        tEnd = tStim + dtStim
        iBegin, iEnd = 0, 0
        for i in range(1, len(t)):
            if t[i - 1] < tStim and t[i] >= tStim:
                iBegin = i
            if t[i - 1] < tEnd and t[i] >= tEnd:
                iEnd = i
        v_ = v[iBegin:iEnd]
        #        steady-state can only be determined for constant
        #        current injection, not for synaptic input;
        #        therefore, use median of this trace as approximation
        vss = np.median(v_)
        return vss

    @staticmethod
    def analyze_single_spike(cell, tVec, thresh):
        '''
        for comparison with in vitro spike shape:
        compute spike height, FWHM (duration), AHP
        '''
        t = np.array(tVec)
        v = np.array(cell.soma.recVList[0])
        vMax = np.max(v)
        height = vMax - thresh
        halfMax = vMax - 0.5 * height
        tBegin, tEnd = 0.0, 0.0
        iEnd = 0
        for i in range(1, len(t)):
            if v[i - 1] < halfMax and v[i] >= halfMax:
                tBegin = t[i]
            if v[i - 1] >= halfMax and v[i] < halfMax:
                tEnd = t[i]
                iEnd = i
        width = tEnd - tBegin
        vMin = np.min(v[iEnd:])
        ahp = thresh - vMin
        return height, width, ahp

    @staticmethod
    def analyze_spike_train():
        pass