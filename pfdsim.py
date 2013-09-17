#!/usr/bin/python

import sys
import math
import random
import numpy as np
import scipy.signal

from rawdata import rawData

class Pulsar:
    def __init__(self, period, dm, snr,duty,smearing=True):
        self.period = period
        self.dm = dm
        self.snr= snr
        self.duty_cycle = duty
        self.smearing = smearing

class Survey:
    def __init__(self, 
                 n_bits = 1,
                 t_samp = 250.E-6,
                 n_samp = 524288,
                 n_freq_chans = 96,
                 freq_chan_width_mhz = 3.,
                 freq_centre = 1352.,
                 n_sub_bands = 8,
                 n_sub_ints = 8,
                 n_prof_bins=128):

        self.n_bits = n_bits
        self.t_samp = t_samp
        self.n_samp = n_samp
        self.n_freq_chans = n_freq_chans
        self.freq_chan_width_mhz = freq_chan_width_mhz
        self.freq_centre = freq_centre
        self.n_sub_bands = n_sub_bands
        self.n_sub_ints = n_sub_ints
        self.n_prof_bins = n_prof_bins

        # first frequency channel 
        self.fch1 = self.freq_centre + \
                    self.freq_chan_width_mhz * self.n_freq_chans/2

class SimSinglePulse:
    """
    Class for single pulse data"""

    def __init__(self, survey, lowDM=0., highDM=1000.): #add pulsar later

        self.t_samp = survey.t_samp
        self.freq_centre = survey.freq_centre
        self.n_freq_chans = survey.n_freq_chans
        self.freq_chan_width_mhz = survey.freq_chan_width_mhz
        self.n_samp = survey.n_samp

        self.rawdata = rawData(n_bits=survey.n_bits,
                          t_samp=survey.t_samp,
                          n_samp=survey.n_samp,
                          n_freq_chans=survey.n_freq_chans,
                          freq_chan_width_mhz=survey.freq_chan_width_mhz,
                          freq_centre=survey.freq_centre)

        self.rawdata.raw_data = np.random.normal(0.,
                                                 1.,
                                                 (self.n_freq_chans, self.n_samp))
        self.digitised = self.rawdata.digitise_array()

        #print np.shape(self.rawdata)
        
        self.singlepulsearray = self.make_single_pulse_array(lowDM, highDM)

    def dump_single_pulse_files(self, thres=4.):

        """loop over singlepulsearray and write a .singlepulse file per dm"""

        for timeseries in self.singlepulsearray:
            # find pulses >threshold (sigma)
            pulses = self._find_pulses_over_thres(timeseries, thres)
            


    def _find_pulses_over_thres(self, timeseries, thres):

        pulses = []

        # list of downfacts
        downfacts = [1,2,3,4,6,9,14,20,30]

        # find pulses above threshold
        for downfact in downfacts:
            # downfactor and normalise
            ts = self._fact_down(timeseries, downfact)
            ts = self._norm(ts)

            # search for peaks
            peakbins = np.flatnonzero(ts>thres)   
            peakvals = ts[peakbins]

            for bin, val in zip(peakbins, peakvals):
                pulses.append((bin, val, downfact))


        # now remove overlapping pulses which are within a 
        # downfact of one another (leave brightest)
        
        #sort pulses by bin
        pulses_sorted = sorted(pulses, key=lambda tup: tup[0])
        keep_pulses = self._remove_pulses_increasing(pulses_sorted)
        # do it again to remove any missed
        keep_pulses = self._remove_pulses_increasing(keep_pulses)

        return pulses

    def _remove_pulses_increasing(self, pulses_sorted):

        keep_pulses=[]

        # loop over all the pulses per pulse
        for p1index, pulse1 in enumerate(pulses_sorted):
            bin1,val1,df1 = pulse1


            local_pulses=[]
            for p2index, pulse2 in enumerate(pulses_sorted):
                bin2, val2, df2 = pulse2
                
                offset = bin1 - bin2
                # don't compare pulse with itself!
                if pulse1 == pulse2:
                    continue
                ### BUG HERE! NEEDS THINKING ABOUT CAREFULLY
                ## IF PULSE1 = PULSE2 and no other pulses,
                ## no pulse would be kept
                if  -df1/2. <= offset <= df1/2.:
                    # so it's within half a decimate factor
                    # add to local_pulses
                    local_pulses.append(pulse2)


            less_bright = 0
            # no local pulses, keep the pulse
            if len(local_pulses) == 0:
                keep_pulses.append(pulse1)

            # otherwise....
            for pulse2 in local_pulses:
                bin2,val2,df2 = pulse2
                
                if val2 > val1:
                    # second pulse is brighter
                    less_bright += 1

            # if less_bright is still zero, keep the pulse
            if less_bright == 0:
                keep_pulses.append(pulse1)

        return keep_pulses



    def _norm(self, ts):
        mn = np.mean(ts)
        std = np.std(ts)

        # normalise the timeseries
        normed = (ts - mn) / std

        return normed

    def _fact_down(self, ts, fact):
        """Down factor the time series by fact"""

        # copying the method used in presto (single_pulse_search.py)
        kernel = np.ones(fact, dtype=np.float32) / np.sqrt(fact)
        overlapped = np.hstack((ts[-fact:], ts, ts[:fact]))
        conv = scipy.signal.convolve(overlapped, kernel, 1)

        # remove edge effects, and you're done
        tsd = conv[fact:-fact]

        return tsd

    def make_single_pulse_array(self, lowDM, highDM):
        """Make an array of time series from lowDM to hiDM"""
        # calculate DMs from lowDM to highDM
        #dmlist = self._calc_dm_sequence(lowDM, highDM)
        dmlist = self._getDMtable(lowDM, highDM)

        t_dm_plane = []
        for dm in dmlist:
            # dedisperse at each dm
            dedisp_array = self.rawdata.dedisperse(dm, self.digitised)

            # squash into time series
            timeseries = self.rawdata.make_timeseries(dedisp_array)
            t_dm_plane.append(timeseries)

            #print dm, np.shape(self.digitised), np.shape(dedisp_array), np.shape(timeseries)



        return t_dm_plane

    def _getDMtable(self, loDM, hiDM, tolerance=1.25, ti=40.):
        """Lina's code to calc dm series"""
        
        dm = 0.
        dm_list=[]
        while dm < hiDM:
            if dm >= loDM:
                dm_list.append(dm)
            teff = self._calc_teff(dm, ti) #assume intrinsic width = 40us

            a = 8.3 * self.freq_chan_width_mhz / (self.freq_centre/1000.)**3
            b = 8.3 * self.n_freq_chans * self.freq_chan_width_mhz / 4. / (self.freq_centre/1000.)**3
            c = tolerance**2 * teff**2 - (self.t_samp * 1e6)**2 - ti*ti

            #print dm, a, b, c
            dm = self._calc_new_dm(dm, a, b, c)

        return dm_list

    def _calc_new_dm(self, dm, a, b, c):

        value = b*b*dm
        value += math.sqrt(-a*a * b*b * dm*dm + a*a*c + b*b*c)
        return value / (a*a + b*b)

    def _calc_teff(self, dm, ti):

        value = 0.
        # include tsamp
        value += self.t_samp * self.t_samp *1.e6*1.e6
        # include intrinsic width
        value += ti * ti
        # include dm smear
        dm_smear = 8.3 * self.freq_chan_width_mhz * dm / (self.freq_centre/1000.)**3

        value += dm_smear * dm_smear

        return math.sqrt(value)

    def _calc_dm_sequence(self, lowDM, highDM):
        """Calculate how to span lowDM to highDM with sensible step sizes"""

        #calc diagonal dm
        diag_dm = self._dm_i(self.n_freq_chans)

        dm = 0.
        dm_list = []
        n_dms = 1
        while dm < highDM:
            # append dm value to list
            if dm >= lowDM:
                dm_list.append(dm)

            # calculate amount to increment by
            increment = int(dm/diag_dm) + 1
            n_dms += increment

            #calc dm value
            dm = self._dm_i(n_dms)

        return dm_list

    def _dm_i(self, i):
        bandwidth = self.n_freq_chans * self.freq_chan_width_mhz
        return 1.205E-7 * (i-1) * self.t_samp * self.freq_centre**3 / (bandwidth)




class SimPfd:
    """
    Class to store candidate plot data.
    This generates the necessary pfd-style candidate from
    simulated raw data
    """

    def __init__(self, pulsar, survey):
        """
        Initialise the pfd - 
                make raw data
                digitise it
                dedisperse it
        """
        # candidates params
        self.period = pulsar.period * 1000.
        self.wint = self.period * pulsar.duty_cycle
        self.dm = pulsar.dm

        # raw data
        rawdata = rawData(n_bits=survey.n_bits,
                               t_samp=survey.t_samp,
                               n_samp=survey.n_samp,
                               n_freq_chans=survey.n_freq_chans,
                               freq_chan_width_mhz=survey.freq_chan_width_mhz,
                               freq_centre=survey.freq_centre)

        rawdata.add_pulsar_isolated(period_s = pulsar.period,
                                     dm = pulsar.dm,
                                     duty_cycle = pulsar.duty_cycle,
                                     snr = pulsar.snr, # this is SNr per pulse
                                     smearing=pulsar.smearing,
                                     noisychannels=10.,
                                     noisysamples=10.)

        digitised = rawdata.digitise_array()

        dedispersed_at_dm = rawdata.dedisperse(self.dm, input_array=digitised)

        # the full, 3(4?)D profile
        self.profs = rawdata.make_pfd_profs(dedispersed_at_dm,
                                                survey.n_sub_bands,
                                                survey.n_sub_ints,
                                                survey.n_prof_bins,
                                                period=pulsar.period)

        self.chi2 = rawdata.calc_redchi2(self.profs)

        # dm curve, pulse profile, subbands, subints
        self.dmchis, self.dms = self._chi2_vs_DM(pulsar,
                                                 survey, 
                                                 rawdata, 
                                                 digitised)
        self.profile = self._profile_1d()
        self.subbands = self._subbands()
        self.subints = self._subints()

    def _chi2_vs_DM(self, pulsar, survey, rawdata, digitised):
        """Loop over DMs and calculate chi2 for each one"""

        if pulsar.period<0.002:
            dmstep = 1.77*pulsar.period
            halfrange = 62.7 * pulsar.period
        elif pulsar.period < 0.05:
            dmstep = 1.29 * pulsar.period
            halfrange = 62.7 * pulsar.period
        elif pulsar.period < 0.5:
            dmstep = .21 * pulsar.period
            halfrange = 24.7 * pulsar.period
        else:
            dmstep = 0.107 * pulsar.period
            halfrange = 24.7 * pulsar.period

        dms = np.arange(pulsar.dm - halfrange, 
                        pulsar.dm + halfrange,
                        dmstep)

        chis=np.array([])
        for dm in dms:
            dedisp = rawdata.dedisperse(dm, input_array = digitised)

            prof = rawdata.make_pfd_profs(dedisp,
                                          survey.n_sub_bands,
                                          survey.n_sub_ints,
                                          survey.n_prof_bins,
                                          period = pulsar.period)

            chis = np.append(chis, rawdata.calc_redchi2(prof))

        return chis, dms

    def _chi2_vs_DM_init(self, pulsar, survey, rawdata, digitised):
        """Loop over DMs and calculate chi2 for each one"""
        dmstep = 1
        if pulsar.period<0.002:
            dmfactor = 2
        else:
            dmfactor = 1

        ndms = 2*dmfactor*survey.n_prof_bins+1
        lodm = pulsar.dm - (ndms-1)*dmstep/2.
        hidm = pulsar.dm + (ndms-1)*dmstep/2.

        dms = np.arange(lodm, hidm, dmstep)
        chis = np.array([])
        for dm in dms:
            dedisp = rawdata.dedisperse(dm, input_array = digitised)
           
            prof = rawdata.make_pfd_profs(dedisp,
                                          survey.n_sub_bands,
                                          survey.n_sub_ints,
                                          survey.n_prof_bins,
                                          period = pulsar.period)
            
            chis = np.append(chis, rawdata.calc_redchi2(prof))

        return chis, dms

    def _profile_1d(self):
        sumprof = self.profs.sum(0).sum(0)
        normprof = sumprof - min(sumprof)
        normprof /= max(normprof)
        return normprof

    def _subbands(self):
        # make subbands and normalise the array
        profs = self.profs.sum(0)
        return self._norm_subs(profs)

    def _subints(self):
        profs = self.profs.sum(1)
        return self._norm_subs(profs)

    def _norm_subs(self, arr):
        arr -= np.min(arr)
        arr /= np.max(arr)
        return arr

if __name__ == '__main__':
    pass
