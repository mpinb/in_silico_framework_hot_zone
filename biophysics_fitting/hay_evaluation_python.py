'''
A Python translation of the evaluation functions used in :cite:t:`Hay_Hill_Schuermann_Markram_Segev_2011`.
This module provides methods to run Hay's stimulus protocols, and evaluate the resulting voltage traces.
'''

from typing import Any
import numpy as np
import six
from .ephys import *
from six import iteritems
from biophysics_fitting.hay_specification import (
    HAY_BAP_DEFINITIONS, 
    HAY_BAC_DEFINITIONS, 
    HAY_STEP1_DEFINITIONS,
    HAY_STEP2_DEFINITIONS,
    HAY_STEP3_DEFINITIONS)
__author__ = 'Arco Bast'
__date__ = '2018-11-08'

# moved to the bottom to resolve circular import
# from .hay_complete_default_setup import get_hay_problem_description, get_hay_objective_names, get_hay_params_pdf

objectives_step = [
    'AI1', 'AI2', 'AI3', 'APh1', 'APh2', 'APh3', 'APw1', 'APw2', 'APw3', 'DI1',
    'DI2', 'ISIcv1', 'ISIcv2', 'ISIcv3', 'TTFS1', 'TTFS2', 'TTFS3', 'fAHPd1',
    'fAHPd2', 'fAHPd3', 'mf1', 'mf2', 'mf3', 'sAHPd1', 'sAHPd2', 'sAHPd3',
    'sAHPt1', 'sAHPt2', 'sAHPt3'
]  # objectives for the step current injection protocol

objectives_BAC = [
    'BAC_APheight', 'BAC_ISI', 'BAC_ahpdepth', 'BAC_caSpike_height',
    'BAC_caSpike_width', 'BAC_spikecount', 'bAP_APheight', 'bAP_APwidth',
    'bAP_att2', 'bAP_att3', 'bAP_spikecount'
]  # objectives for the BAC and bAP protocols


def normalize(raw, mean, std):
    """Normalize a raw value.
    
    Args:
        raw: raw value
        mean: mean value
        std: standard deviation
        
    Returns:
        float: normalized value"""
    return np.mean(np.abs(raw - mean)) / std


def nan_if_error(fun):
    """Wrapper method that returns nan if an error occurs.
    
    Args:
        fun: function to run
        
    Returns:
        function: wrapped function
    """
    def helper(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except:
            return float('nan')

    return helper


class BAC:
    """Evaluate the :math:`BAC` stimulus protocol.
    
    These metrics were introduced by :cite:t:`Hay_Hill_Schuermann_Markram_Segev_2011`, and illustrated in :cite:t:`Guest_Bast_Narayanan_Oberlaender`.
    
    See also:
        :py:meth:`biophysics_fitting.setup_stim.setup_BAC` for more information on the stimulus protocol.
        
    Attributes:
        hot_zone_thresh (float): The threshold for APs in the dendritic voltage trace. Defaults to :math:`-55` mV.
        soma_thresh (float): The threshold for APs in the somatic voltage trace. Defaults to :math:`-30` mV.
        ca_max_after_nth_somatic_spike (int): The number of somatic spikes after which the calcium spike maximum should occur. Defaults to :math:`2`.
        stim_onset (float): The onset of the stimulus (ms). Defaults to :math:`295` ms.
        stim_duration (float): The duration of the stimulus (ms). Defaults to :math:`45` ms.
        repolarization (float): The target repolarization voltage after the stimulus. 
            See :py:meth:`~biophysics_fitting.ephys.BAC_ISI_check_repolarization`.
            Defaults to :math:`-55` mV.
        punish (float): The punishment value in units of :math:`\sigma`. 
            Used as a baseline if the voltage trace cannot be evaluated on a metric (e.g. if it does not contain an AP). 
            Defaults to :math:`250`.
        punish_last_spike_after_deadline (bool): Whether to punish if the last spike is after the deadline. Defaults to ``True``
        punish_minspikenum (int): The minimum number of spikes required for this stimulus protocol.
        punish_returning_to_rest_tolerance (float): The tolerance for returning to rest (:math:`mV`). Defaults to :math:`2 mV`.
        prefix (str): The prefix for the evaluation metric checks. Defaults to an empty string.
        definitions (dict): The empirical means and standard deviations for the evaluation metrics. Defaults to::
            
            {
                'BAC_APheight': ('AP_height', 25.0, 5.0),
                'BAC_ISI': ('BAC_ISI', 9.901, 0.8517),
                'BAC_ahpdepth': ('AHP_depth_abs', -65.0, 4.0),
                'BAC_caSpike_height': ('BAC_caSpike_height', 6.73, 2.54),
                'BAC_caSpike_width': ('BAC_caSpike_width', 37.43, 1.27),
                'BAC_spikecount': ('Spikecount', 3.0, 0.01)
            }
    """

    def __init__(
        self,
        hot_zone_thresh=-55,
        soma_thresh=-30,
        ca_max_after_nth_somatic_spike=2,
        stim_onset=295,
        stim_duration=45,
        repolarization=-55,
        punish=250,
        punish_last_spike_after_deadline=True,
        punish_minspikenum=2,
        punish_returning_to_rest_tolerance=2.,
        punish_max_prestim_dendrite_depo=-50,
        prefix='',
        definitions=None
        ):
        """
        Args:
            hot_zone_thresh (float): The threshold for APs in the dendritic voltage trace. Defaults to :math:`-55` mV.
            soma_thresh (float): The threshold for APs in the somatic voltage trace. Defaults to :math:`-30` mV.
            ca_max_after_nth_somatic_spike (int): The number of somatic spikes after which the calcium spike maximum should occur. Defaults to :math:`2`.
            stim_onset (float): The onset of the stimulus (ms). Defaults to :math:`295` ms.
            stim_duration (float): The duration of the stimulus (ms). Defaults to :math:`45` ms.
            repolarization (float): The target repolarization voltage after the stimulus. 
                See :py:meth:`~biophysics_fitting.ephys.BAC_ISI_check_repolarization`.
                Defaults to :math:`-55` mV.
            punish (float): The punishment value in units of :math:`\sigma`. 
                Used as a baseline if the voltage trace cannot be evaluated on a metric (e.g. if it does not contain an AP). 
                Defaults to :math:`250`.
            punish_last_spike_after_deadline (bool): Whether to punish if the last spike is after the deadline. Defaults to ``True``
            punish_minspikenum (int): The minimum number of spikes required for this stimulus protocol.
            punish_returning_to_rest_tolerance (float): The tolerance for returning to rest (:math:`mV`). Defaults to :math:`2 mV`.
            prefix (str): The prefix for the evaluation metric checks. Defaults to an empty string.
            definitions (dict): The definitions for the evaluation metrics. See also: :py:attr:`definitions`.
        """
        # TODO: punish_max_prestim_dendrite_depo is unused?
        self.hot_zone_thresh = hot_zone_thresh
        self.soma_thresh = soma_thresh
        self.ca_max_after_nth_somatic_spike = ca_max_after_nth_somatic_spike
        self.stim_onset = stim_onset
        self.stim_duration = stim_duration
        self.definitions = definitions if definitions is not None else {
            'BAC_APheight': ('AP_height', 25.0, 5.0),
            'BAC_ISI': ('BAC_ISI', 9.901, 0.8517),
            'BAC_ahpdepth': ('AHP_depth_abs', -65.0, 4.0),
            'BAC_caSpike_height': ('BAC_caSpike_height', 6.73, 2.54),
            'BAC_caSpike_width': ('BAC_caSpike_width', 37.43, 1.27),
            'BAC_spikecount': ('Spikecount', 3.0, 0.01)
        }
        self.repolarization = repolarization
        self.punish = punish
        self.punish_last_spike_after_deadline = punish_last_spike_after_deadline
        self.punish_minspikenum = punish_minspikenum
        self.punish_returning_to_rest_tolerance = punish_returning_to_rest_tolerance
        self.punish_max_prestim_dendrite_depo = punish_max_prestim_dendrite_depo
        self.prefix = prefix

    def get(self, **voltage_traces):
        """Get the full evaluation of the voltage traces for BAC firing.
        
        Args:
            voltage_traces: dictionary with the voltage traces of the soma and dendrite.
            
        Returns:
            dict: dictionary with the evaluation metrics, containing the raw values, normalized values, and checks.
        """
        spikecount = self.BAC_spikecount(voltage_traces)['.raw']
        out = {}
        for evaluation_metric, (_, mean, std) in six.iteritems(self.definitions):
            # special case in the original code were in the case of two somatic spikes
            # 7 is substracted from the mean
            if spikecount == 2 and evaluation_metric == 'BAC_caSpike_width':
                mean = mean - 7.
            evaluation_current_metric = getattr(self, evaluation_metric)(voltage_traces)
            evaluation_current_metric['.normalized'] = normalize(evaluation_current_metric['.raw'], mean, std)
            checks = [v for k, v in six.iteritems(evaluation_current_metric) if 'check' in k]
            if all(checks):
                evaluation_current_metric[''] = evaluation_current_metric['.normalized']
            else:
                evaluation_current_metric[''] = 20.  # *std
            evaluation_current_metric = {
                evaluation_metric + suffix: evaluation_value 
                for suffix, evaluation_value in six.iteritems(evaluation_current_metric)}
            out.update(evaluation_current_metric)
        return self.check(out, voltage_traces)

    def check(self, out, voltage_traces):
        """Check for problems in the voltage trace.
        
        This should be called after evaluating the voltage traces.
        it is e.g. the last step in :py:meth:`get`, where it adds the checks to the output dictionary.
        
        This method checks if the voltage trace:
        
        - has at least 2 APs present.
        - properly returns to rest.
        - has no spikes before stimulus onset (in soma or dendrite).
        - has its last spike before the deadline.
        - there is no dendritic spike before stimulus onset
        
        Args:
            out: dictionary with the evaluation metrics
            voltage_traces: dictionary with the voltage traces of the soma and dendrite.
            
        Returns:
            dict: dictionary with the evaluation metrics, containing the raw values, normalized values, and checks.
        """
        # checking for problems in voltage trace
        t, v = voltage_traces['tVec'], voltage_traces['vList'][0]
        vmax = None  # voltage_traces['vMax']
        
        # basic error check - used if trace e.g. does not spike at all.
        # penalizes low variance, slightly less penalizes higher variance
        err = trace_check_err(
            t,
            v,
            stim_onset=self.stim_onset,
            stim_duration=self.stim_duration,
            punish=self.punish)
        err_flags = trace_check(
            t,
            v,
            stim_onset=self.stim_onset,
            stim_duration=self.stim_duration,
            minspikenum=self.punish_minspikenum,
            soma_threshold=self.soma_thresh,
            returning_to_rest=self.punish_returning_to_rest_tolerance,
            name='BAC',
            vmax=vmax)
        if self.punish_last_spike_after_deadline:
            relevant_err_flags = err_flags
        else:
            import six
            relevant_err_flags = {
                k: v
                for k, v in six.iteritems(err_flags)
                if not 'last_spike_before_deadline' in k
            }
        for evaluation_metric_name in list(self.definitions.keys()):
            if not all(relevant_err_flags.values()):
                out[evaluation_metric_name] = err
            elif out[evaluation_metric_name] > self.punish:
                out[evaluation_metric_name] = self.punish * 0.75
            # change from hay algorithm by arco: if high depolarization anywhere in the dendrite occurs before stimulus, put highest punish value
            if not err_flags['BAC.check_max_prestim_dendrite_depo']:
                out[evaluation_metric_name] = self.punish
        out['BAC.err'] = err
        out.update(err_flags)
        out = {self.prefix + k: out[k] for k in out.keys()}
        return out

    def BAC_spikecount(self, voltage_traces):
        """Get the number of spikes in the somatic voltage trace.
        
        Args:
            voltage_traces (dict): dictionary with the voltage traces of the soma and dendrite.
            
        Returns:
            dict: Dictionary mapping ``".raw"`` to the number of spikes in the somatic voltage trace.
        """
        t, v = voltage_traces['tVec'], voltage_traces['vList'][0]
        return {
            '.raw': nan_if_error(spike_count)(t, v, thresh=self.soma_thresh)
        }

    def BAC_APheight(self, voltage_traces):
        """Get the height of the first action potential in the somatic voltage trace.
        
        Args:
            voltage_traces (dict): dictionary with the voltage traces of the soma and dendrite.
            
        Returns:
            dict: Dictionary mapping ``".check_1AP"`` to whether there is at least one action potential in the somatic voltage trace,
                and ``".raw"`` to the height of the first action potential in the somatic voltage trace.
                
        See also:
            :py:func:`AP_height`
        """
        t, v = voltage_traces['tVec'], voltage_traces['vList'][0]
        return {
            '.check_1AP':
                nan_if_error(AP_height_check_1AP)(t, v,
                                                  thresh=self.soma_thresh),
            '.raw':
                nan_if_error(AP_height)(t, v, thresh=self.soma_thresh)
        }

    def BAC_ISI(self, voltage_traces):
        """Get the interspike interval in the somatic voltage trace.
        
        Args:
            voltage_traces (dict): dictionary with the voltage traces of the soma and dendrite.
            
        Returns:
            dict: Dictionary mapping ``".check_2_or_3_APs"`` to whether there are 2 or 3 action potentials in the somatic voltage trace,
                ``".check_repolarization"`` to whether the voltage trace repolarizes after the stimulus,
                and ``".raw"`` to the interspike interval in the somatic voltage trace.
                
        See also:
            :py:func:`BAC_ISI`
        """
        t, v = voltage_traces['tVec'], voltage_traces['vList'][0]
        stim_end = self.stim_onset + self.stim_duration
        r = self.repolarization
        return {
            '.check_2_or_3_APs':
                nan_if_error(BAC_ISI_check_2_or_3_APs)(t,
                                                       v,
                                                       thresh=self.soma_thresh),
            '.check_repolarization':
                nan_if_error(BAC_ISI_check_repolarization)(t,
                                                           v,
                                                           stim_end=stim_end,
                                                           repolarization=r),
            '.raw':
                nan_if_error(BAC_ISI)(t, v, thresh=self.soma_thresh)
        }

    def BAC_ahpdepth(self, voltage_traces):
        """Get the afterhyperpolarization depth in the somatic voltage trace.
        
        Args:
            voltage_traces (dict): dictionary with the voltage traces of the soma and dendrite.
            
        Returns:
            dict: Dictionary mapping ``".check_2AP"`` to whether there are 2 action potentials in the somatic voltage trace,
                and ``".raw"`` to the afterhyperpolarization depth in the somatic voltage trace.
                
        See also:
            :py:func:`AHP_depth_abs`
        """
        t, v = voltage_traces['tVec'], voltage_traces['vList'][0]
        return {
            '.check_2AP':
                nan_if_error(AHP_depth_abs_check_2AP)(t,
                                                      v,
                                                      thresh=self.soma_thresh),
            '.raw':
                nan_if_error(AHP_depth_abs)(t, v, thresh=self.soma_thresh)
        }

    def BAC_caSpike_height(self, voltage_traces):
        """Get the height of the calcium spike in the dendritic voltage trace.
        
        Args:
            voltage_traces (dict): dictionary with the voltage traces of the soma and dendrite.
            
        Returns:
            dict: Dictionary mapping ``".check_1_Ca_AP"`` to whether there is at least one calcium spike in the dendritic voltage trace,
                ``".check_>=2_Na_AP"`` to whether there are at least two action potentials in the somatic voltage trace,
                ``".check_ca_max_after_nth_somatic_spike"`` to whether the calcium spike occurs after the nth somatic spike,
                and ``".raw"`` to the height of the calcium spike in the dendritic voltage trace.
                
        See also:
            :py:func:`BAC_caSpike_height`
        """
        t, v = voltage_traces['tVec'], voltage_traces['vList'][0]
        v_dend = voltage_traces['vList'][1]
        tend = self.stim_duration + self.stim_onset
        return {'.check_1_Ca_AP': nan_if_error(BAC_caSpike_height_check_1_Ca_AP)(t,v,v_dend,thresh=self.hot_zone_thresh),
                '.check_>=2_Na_AP': nan_if_error(BAC_caSpike_height_check_gt2_Na_spikes)(t,v,v_dend,thresh=self.soma_thresh),
                '.check_ca_max_after_nth_somatic_spike':  \
                    nan_if_error(BAC_caSpike_height_check_Ca_spikes_after_Na_spike)(t,v,v_dend,
                                                                      n=self.ca_max_after_nth_somatic_spike,
                                                                      thresh=self.soma_thresh),
                '.raw': nan_if_error(BAC_caSpike_height)(t,v,v_dend,ca_thresh=self.hot_zone_thresh,tstim=self.stim_onset)}

    def BAC_caSpike_width(self, voltage_traces):
        """Get the width of the calcium spike in the dendritic voltage trace.
        
        Args:
            voltage_traces (dict): dictionary with the voltage traces of the soma and dendrite.
            
        Returns:
            dict: Dictionary mapping ``".check_1_Ca_AP"`` to whether there is at least one calcium spike in the dendritic voltage trace,
                and ``".raw"`` to the width of the calcium spike in the dendritic voltage trace.
                
        See also:
            :py:func:`BAC_caSpike_width`
        """
        t, v = voltage_traces['tVec'], voltage_traces['vList'][0]
        v_dend = voltage_traces['vList'][1]
        return {
            '.check_1_Ca_AP':
                nan_if_error(BAC_caSpike_width_check_1_Ca_AP)
                (t, v, v_dend, thresh=self.hot_zone_thresh),
            '.raw':
                nan_if_error(BAC_caSpike_width)(t,
                                                v,
                                                v_dend,
                                                thresh=self.hot_zone_thresh)
        }


class bAP:
    """Evaluate the :math:`bAP` stimulus protocol.
    
    These metrics were introduced by :cite:t:`Hay_Hill_Schuermann_Markram_Segev_2011`, and illustrated in :cite:t:`Guest_Bast_Narayanan_Oberlaender`.
    
    See also:
        :py:meth:`biophysics_fitting.setup_stim.setup_bAP` for more information on the stimulus protocol.
        
    Attributes:
        soma_thresh (float): The threshold for APs in the somatic voltage trace. Defaults to :math:`-30` mV.
        stim_onset (float): The onset of the stimulus (ms). Defaults to :math:`295` ms.
        stim_duration (float): The duration of the stimulus (ms). Defaults to :math:`5` ms.
        bAP_thresh (float): The threshold for the backpropagating action potential. Defaults to :math:`+2` mV.
        punish (float): The punishment value in units of :math:`\sigma`. 
            Used as a baseline if the voltage trace cannot be evaluated on a metric (e.g. if it does not contain an AP). 
            Defaults to :math:`250 \sigma`.
        punish_last_spike_after_deadline (bool): Whether to punish if the last spike is after the deadline. Defaults to ``True``
        punish_minspikenum (int): The minimum number of spikes required for this stimulus protocol.
        punish_returning_to_rest_tolerance (float): The tolerance for returning to rest (:math:`mV`). Defaults to :math:`2 mV`.
        definitions (dict): The empirical means and standard deviations for the evaluation metrics. Defaults to::
            
            {
                'bAP_APheight': ('AP_height', 25.0, 5.0),
                'bAP_APwidth': ('AP_width', 2.0, 0.5),
                'bAP_att2': ('BPAPatt2', 45.0, 10.0),
                'bAP_att3': ('BPAPatt3', 36.0, 9.33),
                'bAP_spikecount': ('Spikecount', 1.0, 0.01)
            }
    """
    def __init__(
        self,
        soma_thresh=-30,
        stim_onset=295,
        stim_duration=5,
        bAP_thresh='+2mV',
        punish=250.,
        punish_last_spike_after_deadline=True,
        punish_minspikenum=1,
        punish_returning_to_rest_tolerance=2.,
        definitions={
            'bAP_APheight': ('AP_height', 25.0, 5.0),
            'bAP_APwidth': ('AP_width', 2.0, 0.5),
            'bAP_att2': ('BPAPatt2', 45.0, 10.0),
            'bAP_att3': ('BPAPatt3', 36.0, 9.33),
            'bAP_spikecount': ('Spikecount', 1.0, 0.01)
        }):
        """
        Args:
            soma_thresh (float): The threshold for APs in the somatic voltage trace. Defaults to :math:`-30\ mV`.
            stim_onset (float): The onset of the stimulus (:math:`ms`). Defaults to :math:`295\ ms`.
            stim_duration (float): The duration of the stimulus (:math:`ms`). Defaults to :math:`5\ ms`.
            bAP_thresh (float): The threshold for the backpropagating action potential. Defaults to :math:`+2\ mV`.
            punish (float): The punishment value in units of :math:`\sigma`. 
                Used as a baseline if the voltage trace cannot be evaluated on a metric (e.g. if it does not contain an AP). 
                Defaults to :math:`250 \sigma`.
            punish_last_spike_after_deadline (bool): Whether to punish if the last spike is after the deadline. Defaults to ``True``
            punish_minspikenum (int): The minimum number of spikes required for this stimulus protocol. Defaults to :math:`1`.
            punish_returning_to_rest_tolerance (float): The tolerance for returning to rest (:math:`mV`). Defaults to :math:`2 mV`.
            definitions (dict): The definitions for the evaluation metrics. See also: :py:attr:`definitions`.
        """
        self.soma_thresh = soma_thresh
        self.stim_onset = stim_onset
        self.stim_duration = stim_duration
        self.definitions = definitions
        self.bAP_thresh = bAP_thresh
        self.punish = punish
        self.punish_last_spike_after_deadline = punish_last_spike_after_deadline
        self.punish_minspikenum = punish_minspikenum
        self.punish_returning_to_rest_tolerance = punish_returning_to_rest_tolerance

    def get(self, **voltage_traces):
        """Get the full evaluation of the voltage traces for bAP firing.
        
        Args:
            voltage_traces: dictionary with the voltage traces of the soma and dendrite.
            
        Returns:
            dict: dictionary with the evaluation metrics, containing the raw values, normalized values, and checks.
        """
        out = {}
        for name, (_, mean, std) in iteritems(self.definitions):
            out_current = getattr(self, name)(voltage_traces)
            out_current['.normalized'] = normalize(out_current['.raw'], mean, std)
            checks = [v for k, v in iteritems(out_current) if 'check' in k]
            if all(checks):
                out_current[''] = out_current['.normalized']
            else:
                out_current[''] = 20  # *std
            out_current = {name + k: v for k, v in iteritems(out_current)}
            out.update(out_current)
        return self.check(out, voltage_traces)

    def check(self, out, voltage_traces):
        """Check for problems in the voltage trace.
        
        This should be called after evaluating the voltage traces.
        it is e.g. the last step in :py:meth:`get`, where it adds the checks to the output dictionary.
        
        This method checks if the voltage trace:
        
        - has at least 1 AP present.
        - properly returns to rest.
        - has no spikes before stimulus onset (in soma or dendrite).
        - has its last spike before the deadline.
        
        Args:
            out: dictionary with the evaluation metrics
            voltage_traces: dictionary with the voltage traces of the soma and dendrite.
            
        Returns:
            dict: dictionary with the evaluation metrics, containing the raw values, normalized values, and checks.
        """
        # checking for problems in voltage trace
        t, v = voltage_traces['tVec'], voltage_traces['vList'][0]
        vmax = None  # voltage_traces['vMax']
        err = trace_check_err(
            t,
            v,
            stim_onset=self.stim_onset,
            stim_duration=self.stim_duration,
            punish=self.punish)
        err_flags = trace_check(
            t,
            v,
            stim_onset=self.stim_onset,
            stim_duration=self.stim_duration,
            minspikenum=self.punish_minspikenum,
            soma_threshold=self.soma_thresh,
            returning_to_rest=self.punish_returning_to_rest_tolerance,
            name='bAP',
            vmax=vmax)
        if self.punish_last_spike_after_deadline:
            relevant_err_flags = err_flags
        else:
            relevant_err_flags = {
                k: v
                for k, v in iteritems(err_flags)
                if not 'last_spike_before_deadline' in k
            }
        for name in list(self.definitions.keys()):
            if not all(relevant_err_flags.values()):
                out[name] = err
            elif out[name] > self.punish:
                out[name] = self.punish * 0.75
            if not err_flags['bAP.check_max_prestim_dendrite_depo']:
                out[name] = self.punish
        out['bAP.err'] = err
        out.update(err_flags)
        return out

    def bAP_APheight(self, voltage_traces):
        """Get the height of the first action potential in the somatic voltage trace.
        
        Args:
            voltage_traces (dict): dictionary with the voltage traces of the soma and dendrite.
            
        Returns:
            dict: Dictionary mapping ``".check_1AP"`` to whether there is at least one action potential in the somatic voltage trace,
            and ``".raw"`` to the height of the first action potential in the somatic voltage trace.
        
        See also:
            :py:meth:`~biophysics_fitting.ephys.APheight`
        """
        t, v = voltage_traces['tVec'], voltage_traces['vList'][0]
        return {
            '.check_1AP':
                nan_if_error(AP_height_check_1AP)(t, v, thresh=self.soma_thresh),
            '.raw':
                nan_if_error(AP_height)(t, v, thresh=self.soma_thresh)
        }

    def bAP_APwidth(self, voltage_traces):
        """Get the width of the first action potential in the somatic voltage trace.
        
        Args:
            voltage_traces (dict): dictionary with the voltage traces of the soma and dendrite.
            
        Returns:
            dict: Dictionary mapping ``".check_1AP"`` to whether there is at least one action potential in the somatic voltage trace,
            and ``".raw"`` to the width of the first action potential in the somatic voltage trace.
        
        See also:
            :py:meth:`~biophysics_fitting.ephys.APwidth`
        """
        t, v = voltage_traces['tVec'], voltage_traces['vList'][0]
        return {
            '.check_1AP':
                nan_if_error(AP_width_check_1AP)(t, v, thresh=self.soma_thresh),
            '.raw':
                nan_if_error(AP_width)(t, v, thresh=self.soma_thresh)
        }

    def bAP_spikecount(self, voltage_traces):
        """Get the number of spikes in the somatic voltage trace.
        
        Args:
            voltage_traces (dict): dictionary with the voltage traces of the soma and dendrite.
            
        Returns:
            dict: Dictionary mapping ``".raw"`` to the number of spikes in the somatic voltage trace.
        
        See also:
            :py:meth:`~biophysics_fitting.ephys.spike_count`
        """
        t, v = voltage_traces['tVec'], voltage_traces['vList'][0]
        return {
            '.raw': nan_if_error(spike_count)(t, v, thresh=self.soma_thresh)
        }

    def _bAP_att(self, voltage_traces, _n=1):
        """Get the backpropagating action potential attenuation.
        
        Args:
            voltage_traces (dict): dictionary with the voltage traces of the soma and dendrite.
            _n (int): 
                The index of the dendritic voltage trace.
                Index :math:`0` is the somatic voltage trace, 
                index :math:`1` is the first dendritic voltage trace, and
                index :math:`2` is the second dendritic voltage trace.
                Defaults to :math:`1` i.e. the first pipette in the dendrite.
                
        Returns:
            dict: Dictionary mapping ``".raw"`` to the backpropagating action potential attenuation.
            
        See also:
            :py:meth:`~biophysics_fitting.ephys.BPAPatt`
        """
        t, v = voltage_traces['tVec'], voltage_traces['vList'][0]
        v_dend = voltage_traces['vList'][_n]
        return {
            '.raw':
                nan_if_error(BPAPatt)(t, v_dend, self.bAP_thresh,
                                      self.stim_onset),
            '.check_1_AP':
                nan_if_error(BPAPatt_check_1_AP)(t, v, thresh=self.soma_thresh),
            '.check_relative_height':
                nan_if_error(BPAPatt_check_relative_height)
                (t, v, v_dend, self.bAP_thresh, self.stim_onset)
        }

    def bAP_att2(self, voltage_traces):
        """Get the backpropagating action potential attenuation between the soma and first dendritic pipette location.
        
        Args:
            voltage_traces (dict): dictionary with the voltage traces of the soma and dendrite.
        
        Returns:
            dict: Dictionary mapping ``".raw"`` to the backpropagating action potential attenuation.
            
        See also:
            :py:meth:`~biophysics_fitting.ephys.BPAPatt`    
        """
        return self._bAP_att(voltage_traces, _n=1)

    def bAP_att3(self, voltage_traces):
        """Get the backpropagating action potential attenuation between the soma and second dendritic pipette location.
        
        Args:
            voltage_traces (dict): dictionary with the voltage traces of the soma and dendrite.
            
        Returns:
            dict: Dictionary mapping ``".raw"`` to the backpropagating action potential attenuation.
            
        See also:
            :py:meth:`~biophysics_fitting.ephys.BPAPatt`
        """
        return self._bAP_att(voltage_traces, _n=2)

class _Step:
    """Template class for evaluating step current injections."""
    def __init__(
        self,
        soma_thresh=-30,
        stim_onset=700,
        stim_duration=2000,
        punish=250.,
        punish_last_spike_after_deadline=True,
        punish_minspikenum=5,
        punish_returning_to_rest_tolerance=2.,
        definitions=None,
        name='StepTemplate',
        step_index=0
        ):

        assert definitions is not None, "The Step class is a template, and must be filled with mean and st values (definitions), depending on the current amplitude. Refer to bipohysics_fitting.hay_specification for the definitions."
        self.soma_thresh = soma_thresh
        self.stim_onset = stim_onset
        self.stim_duration = stim_duration
        self.definitions = definitions
        self.punish = punish
        self.punish_last_spike_after_deadline = punish_last_spike_after_deadline
        self.punish_minspikenum = punish_minspikenum
        self.punish_returning_to_rest_tolerance = punish_returning_to_rest_tolerance
        self.name = name
        self.step_index = step_index

    def get(self, **voltage_traces):
        out = {}
        for name, (_, mean, std) in iteritems(self.definitions):
            out_current = getattr(self, name)(voltage_traces)
            out_current['.normalized'] = normalize(out_current['.raw'], mean, std)
            checks = [v for k, v in iteritems(out_current) if 'check' in k]
            if all(checks):
                out_current[''] = out_current['.normalized']
            else:
                out_current[''] = 20  # *std
            out_current = {name + k: v for k, v in iteritems(out_current)}
            out.update(out_current)
        return self.check(out, voltage_traces)

    def check(self, out, voltage_traces):
        # checking for problems in voltage trace
        t, v = voltage_traces['tVec'], voltage_traces['vList'][0]
        vmax = None  # voltage_traces['vMax']
        err = trace_check_err(
            t,
            v,
            stim_onset=self.stim_onset,
            stim_duration=self.stim_duration,
            punish=self.punish)
        err_flags = trace_check(
            t,
            v,
            stim_onset=self.stim_onset,
            stim_duration=self.stim_duration,
            minspikenum=self.punish_minspikenum,
            soma_threshold=self.soma_thresh,
            returning_to_rest=self.punish_returning_to_rest_tolerance,
            name=self.name,
            vmax=vmax)
        for name in list(self.definitions.keys()):
            if not all(err_flags.values()):
                out[name] = err
            elif out[name] > self.punish:
                out[name] = self.punish * 0.75
        out['{}.err'.format(self.name)] = err
        out.update(err_flags)
        return out

    def mf(self, voltage_traces):
        t, v = voltage_traces['tVec'], voltage_traces['vList'][0]
        return {
            '.check_1AP': nan_if_error(spike_count)(t, v, thresh=self.soma_thresh),
            '.raw': nan_if_error(STEP_mean_frequency)(t, v, self.stim_duration, thresh=self.soma_thresh)
        }
        
    def AI(self, voltage_traces):
        t, v = voltage_traces['tVec'], voltage_traces['vList'][0]
        return {
            '.check_2ISI': nan_if_error(spike_count)(t, v, thresh=self.soma_thresh),
            '.raw': nan_if_error(STEP_adaptation_index)(t, v, stim_end=self.stim_onset + self.stim_duration, thresh=self.soma_thresh)
        }
        
    def ISIcv(self, voltage_traces):
        t, v = voltage_traces['tVec'], voltage_traces['vList'][0]
        return {
            '.check_2ISI': nan_if_error(spike_count)(t, v, thresh=self.soma_thresh),
            '.raw': nan_if_error(STEP_coef_var)(t, v, stim_end = self.stim_onset+self.stim_duration, thresh=self.soma_thresh)
        }
        
    def DI(self, voltage_traces):
        t, v = voltage_traces['tVec'], voltage_traces['vList'][0]
        return {
            '.check_2ISI': nan_if_error(spike_count)(t, v, thresh=self.soma_thresh),
            '.raw': nan_if_error(STEP_initial_ISI)(t, v, thresh=self.soma_thresh)
        }
        
    def TTFS(self, voltage_traces):
        t, v = voltage_traces['tVec'], voltage_traces['vList'][0]
        return {
            '.check_1AP': nan_if_error(spike_count)(t, v, thresh=self.soma_thresh),
            '.raw': nan_if_error(STEP_time_to_first_spike)(t, v, self.stim_onset, thresh=self.soma_thresh)
        }
        
    def AHP_depth_abs(self, voltage_traces):
        t, v = voltage_traces['tVec'], voltage_traces['vList'][0]
        return {
            '.check_2AP': nan_if_error(AHP_depth_abs_check_2AP)(t, v, thresh=self.soma_thresh),
            '.raw': nan_if_error(AHP_depth_abs)(t, v, thresh=self.soma_thresh)
        }
        
    def APh(self, voltage_traces):
        t, v = voltage_traces['tVec'], voltage_traces['vList'][0]
        return {
            '.check_1AP':
                nan_if_error(AP_height_check_1AP)(t, v, thresh=self.soma_thresh),
            '.raw':
                nan_if_error(AP_height)(t, v, thresh=self.soma_thresh)
        }
        
    def fAHPd(self, voltage_traces):
        t, v = voltage_traces['tVec'], voltage_traces['vList'][0]
        return {
            '.check_2AP':
                nan_if_error(AHP_depth_abs_check_2AP)(t, v, thresh=self.soma_thresh),
            '.raw':
                nan_if_error(STEP_fast_ahp_depth)(t, v, thresh=self.soma_thresh)
        }
    
    def sAHPd(self, voltage_traces):
        t, v = voltage_traces['tVec'], voltage_traces['vList'][0]
        return {
            '.check_2AP':
                nan_if_error(AHP_depth_abs_check_2AP)(t, v, thresh=self.soma_thresh),
            '.raw':
                nan_if_error(STEP_slow_ahp_depth)(t, v, thresh=self.soma_thresh)
        }
    
    def sAHPt(self, voltage_traces):
        t, v = voltage_traces['tVec'], voltage_traces['vList'][0]
        return {
            '.check_2AP':
                nan_if_error(AHP_depth_abs_check_2AP)(t, v, thresh=self.soma_thresh),
            '.raw':
                nan_if_error(STEP_slow_ahp_time)(t, v, thresh=self.soma_thresh)
        }

    def APw(self, voltage_traces):
        t, v = voltage_traces['tVec'], voltage_traces['vList'][0]
        return {
            '.check_1AP':
                nan_if_error(AP_width_check_1AP)(t, v, thresh=self.soma_thresh),
            '.raw':
                nan_if_error(AP_width)(t, v, thresh=self.soma_thresh)
        }
        
    def __getattr__(self, name):
        """Suffix the evaluation objective with the step index."""
        assert hasattr(self, name.rstrip(str(self.step_index))), f"Attribute {name} not found in {self.name}"
        return object.__getattribute__(self, name.rstrip(str(self.step_index)))

class StepOne(_Step):
    def __init__(self):
        super().__init__(definitions=HAY_STEP1_DEFINITIONS, name='StepOne', step_index='1')
        
class StepTwo(_Step):
    def __init__(self):
        super().__init__(definitions=HAY_STEP2_DEFINITIONS, name='StepTwo', step_index='2')
        
class StepThree(_Step):
    def __init__(self):
        super().__init__(definitions=HAY_STEP3_DEFINITIONS, name='StepThree', step_index='3')

def get_evaluate_bAP(**kwargs):
    """Get the evaluation function for the :math:`bAP` stimulus protocol.
    
    Initializes a :py:class:`bAP` object with the given keyword arguments, 
    and returns a function that evaluates the voltage traces.
    
    Args:
        kwargs: keyword arguments for the :py:class:`bAP` object.
        
    Returns:
        Callable: function that evaluates the voltage traces.
    """
    bap = bAP(**kwargs)

    def fun(**kwargs):
        return bap.get(**kwargs)

    return fun


def get_evaluate_BAC(**kwargs):
    """Get the evaluation function for the :math:`BAC` stimulus protocol.
    
    Initializes a :py:class:`BAC` object with the given keyword arguments,
    and returns a function that evaluates the voltage traces.
    
    Args:
        kwargs: keyword arguments for the :py:class:`BAC` object.
        
    Returns:
        Callable: function that evaluates the voltage traces.
    """
    bac = BAC(**kwargs)

    def fun(**kwargs):
        return bac.get(**kwargs)

    return fun

def get_evaluate_StepOne(**kwargs):
    step = StepOne(**kwargs)

    def fun(**kwargs):
        return step.get(**kwargs)

    return fun

def get_evaluate_StepTwo(**kwargs):
    step = StepTwo(**kwargs)

    def fun(**kwargs):
        return step.get(**kwargs)

    return fun

def get_evaluate_StepThree(**kwargs):
    step = StepThree(**kwargs)

    def fun(**kwargs):
        return step.get(**kwargs)

    return fun