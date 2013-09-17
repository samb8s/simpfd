#!/usr/bin/python

import sys

import math
import random as rand

import numpy as np

class rawData:
    """ Class tp contain typical parameters of a raw data set """
    def __init__(self,
                 n_bits = 1,
                 t_samp = 250.E-6,
                 n_samp = 524288,
                 n_freq_chans = 96,
                 freq_chan_width_mhz = 3.,
                 freq_centre = 1352.,
                 dig_max=+4.,
                 dig_min=-4.
                 ):

        """ Initialise the parameters"""
        self.n_bits = n_bits
        self.t_samp = t_samp
        self.n_samp = n_samp
        self.n_freq_chans = n_freq_chans
        self.freq_chan_width_mhz = freq_chan_width_mhz
        self.freq_centre = freq_centre

        # first frequency channel 
        self.fch1 = self.freq_centre + \
                    self.freq_chan_width_mhz * self.n_freq_chans/2

        self.raw_data = np.array([])

        # max and min for digitising
        self.digit_max = dig_max
        self.digit_min = dig_min

    def _initialise_raw_data(self):
        # making the raw data initialise with index values - can then use these
        # values when filling the array -> faster
        freq_chan = np.arange(0, self.n_samp, dtype = int)

        for i in range(self.n_freq_chans):
            self.raw_data = np.append(self.raw_data, freq_chan)

        self.raw_data = self.raw_data.reshape(self.n_freq_chans, -1)

    def _form_index_array(self):
        # making the raw data initialise with index values - can then use these
        # values when filling the array -> faster
        new_data = np.array([])
        freq_chan = np.arange(0, self.n_samp, dtype = int)

        for i in range(self.n_freq_chans):
            new_data = np.append(new_data, freq_chan)

        return new_data.reshape(self.n_freq_chans, -1)



    def _dm_delay(self,f1,f2,dm):
        """ Calculate the time delay between f1 and f2 for pulse with DM=dm"""

        return(4148.741601*((1.0/f1/f1)-(1.0/f2/f2))*dm)

    def _smear(self, dc, dm, p):
        """Smear a pulse with the sampling time and dm across freq chan"""

        tdm=8.3e3 * dm * self.freq_chan_width_mhz / self.fch1**3
        dc *= p
        dc = math.sqrt(tdm * tdm+ self.t_samp * self.t_samp + dc * dc)

        return dc / p

    def add_pulsar_isolated(self,
                   period_s=.1,
                   dm = 100.,
                   duty_cycle = 0.05,
                   snr = 1.,
                   smearing=False,
                   noisychannels=0.,
                   noisysamples=0.
                   ):

        """ Add a pulsar into the raw_data"""
        
        if smearing:
            duty_cycle = self._smear(duty_cycle, dm, period_s)
                
        # rising and falling phase
        rising = 0.5 - duty_cycle * 0.5
        falling = 0.5 + duty_cycle * 0.5

        # modify SNR to per channel
        snr = snr / math.sqrt(float(self.n_freq_chans))
        # and then divide by sqrt(n pulses)
        npulses = int(self.n_samp * self.t_samp / period_s)
        snr = snr / math.sqrt(npulses)
        if snr > 1.0:
            self.digit_max *= snr

        # initialise the raw data array
        index_array = self._form_index_array()

        # get indices of values which are > rising or <falling
        # (i.e. inside the pulse window)
        pulse_phase = np.modf(index_array*self.t_samp/period_s)[0]
        index1 = pulse_phase>=rising
        index2 = pulse_phase<=falling

        # array of random noise (mean = 0, variance = 1)
        self.raw_data = np.random.normal(0.,1.,(self.n_freq_chans, self.n_samp))

        # add the pulse to the relevant bins
        self.raw_data[index1 & index2] += snr 

        # add some noisy channels or samples
        
        self.addnoise(self.raw_data, noisychannels, noisysamples)

        # need to add delay to pulses at different frequencies....
        for chan_indx, chan in enumerate(self.raw_data):
            fch2 = self.fch1 - chan_indx * self.freq_chan_width_mhz
            delay = self._dm_delay(self.fch1,
                                   fch2,
                                   dm)
            # convert time delay to number of bins (rounding)
            nshift = int(np.rint(delay/self.t_samp))
            # then roll the chan that many bins
            self.raw_data[chan_indx] = np.roll(chan, nshift)

    def addnoise(self, input_array, pc_chans, pc_samps):
        """Make pc percent (on avrg) of samps or channels noisy/blank"""
        pc_chans /= 100.
        pc_samps /= 100.

        ###### CAN DO THIS BETTER!!! USING ARRAYS (get a list of 
        # X% random array indexes, then apply to those in one line

        for ii in range(self.n_freq_chans):
            if rand.random()<pc_chans:
                if rand.random()>0.5:
                    # make noisy
                    input_array[ii] = np.random.normal(0.,3.,(self.n_samp))
                else:
                    # make blank
                    input_array[ii] = np.zeros(self.n_samp, float)

        for jj in range(self.n_samp): 
            if rand.random()<pc_samps:
                if rand.random()>0.5:
                    # make noisy
                    input_array[:,jj] = np.random.normal(0.,
                                                         3.,
                                                         (self.n_freq_chans))
                else:
                    # make blank
                    input_array[:,jj] = np.zeros(self.n_freq_chans, float)

    def add_noise_array(self, input_array, pc_chans, pc_samps):
        """Add noise to PC percent channels/samples"""
        pc_chans /=100.
        pc_samps /=100.

        # channels are first index, samples are second index
        n_noisy_chans = int(pc_chans * input_array.shape[0])
        n_noisy_samps = int(pc_samps * input_array.shape[1])

        # generate list of random indexes to noisify
        chan_index = np.random.random_integers(0,
                                               input_array.shape[0],
                                               n_noisy_chans)
        samp_index = np.random.random_integers(0,
                                               input_array.shape[1],
                                               n_noisy_samps)

        # use the randomly-picked indices to noisify channels/samples
        for ii in chan_index:
            if rand.random()>0.5:
                # make noisy
                input_array[ii] = np.random.normal(0.,3.,(self.n_samp))
            else:
                # make blank
                input_array[ii] = np.zeros(self.n_samp, float)

        for jj in samp_index:
            if rand.random()>0.5:
                # make noisy
                input_array[:,jj] = np.random.normal(0.,
                                                     3.,
                                                     (self.n_freq_chans))
            else:
                # make blank
                input_array[:,jj] = np.zeros(self.n_freq_chans, float)

    def dedisperse(self, dm, input_array=None):
        """Dedisperse the self.raw_data array"""
        if input_array is None:
            input_array = self.raw_data

        output_array = np.copy(input_array)

        for chan_indx, chan in enumerate(output_array):
            fch2 = self.fch1 - chan_indx * self.freq_chan_width_mhz
            delay = self._dm_delay(self.fch1,
                                   fch2,
                                   dm)
            # convert to a number of bins (rounding)
            nshift = int(np.rint(delay/self.t_samp))
            output_array[chan_indx] = np.roll(chan, -nshift)

        return output_array
        
    def digitise_array(self):
        #digit_max = self.raw_data.max()
        #digit_min = self.raw_data.min()
        
        digitised_array = self.raw_data.copy()

        # get indices of cells outside specified range
        index1 = digitised_array <=self.digit_min
        index2 = digitised_array >= self.digit_max

        # replace the values
        digitised_array[index1] = self.digit_min
        digitised_array[index2] = self.digit_max

        # "digitise" any other values
        bit_fac = 2.0**self.n_bits - 1.0
        delta = self.digit_max - self.digit_min

        digitised_array = np.rint(digitised_array-self.digit_min) * bit_fac/delta

        return digitised_array.astype(int)

    def make_timeseries(self, array_name):
        return array_name.sum(axis=0)

    def fft(self, timeseries):
        return np.abs(np.fft.fft(timeseries))

    def make_pfd_profs(self, input_array, nband, nint, nprofbin, period):
        
        nfchan = np.shape(input_array)[0]
        ntchan = np.shape(input_array)[1]

        if nfchan % nband:
            print 'Number of bins, {0}, is not integer multiple of number'\
                    'of sub-bins, {1}'.format(nfchan, nband)
            sys.exit()
        if ntchan % nint:
            print 'Number of bins, {0}, is not integer multiple of number'\
                    'of sub-bins, {1}'.format(ntchan, nint)
            sys.exit()

        n_tchan_in_subint = ntchan/nint
        n_fchan_in_subband = nfchan/nband

        profs = np.zeros((nint, nband, nprofbin), dtype='d')

        for tchan in range(nint):
            #get the subint range
            tindx = tchan*n_tchan_in_subint
            tslice = input_array[:,tindx:tindx+n_tchan_in_subint]
            
            for fchan in range(nband):
                # make subband in the subint
                findx = fchan * n_fchan_in_subband
                fslice = tslice[findx:findx+n_fchan_in_subband]

                # fold at period
                folded = self.fold_subs(fslice, period)
                # sum (ie collapse to one fband in the subint)
                folded = folded.sum(axis=0)
                #rebin to n_prof_bin
                folded = self.reduce_n_bins(folded, nprofbin)
                #folded = self.alt_bin_reduce(folded, nprofbin)
                # then add to the profs array
                profs[tchan][fchan] = folded

        # rotate the subints to account for drift in phase of pulse
        # where subint length and period are not integer-multiples 
        # of one another        
        n_phase_bins_roll = self._rotate_phase_bins(period,
                                                    nint,
                                                    nprofbin)
        for i, tchan in enumerate(profs):
            n_roll = int(i * n_phase_bins_roll)
            profs[i] = np.roll(tchan, -n_roll)
        
        return profs

    def _rotate_phase_bins(self, period, nsubint, nprofbin):
        nsamps = self.n_samp / nsubint
        len_subint = nsamps * self.t_samp
        # delta_t_1 is time left over after max n pulses have occurred in the
        # subint
        delta_t_1 = len_subint - math.floor(len_subint/period) * period

        # delta_t_2 is then the delay of pulse in next subint
        delta_t_2 = period - delta_t_1

        # phase shift 
        delta_phi = delta_t_2/period
        return delta_phi * nprofbin
        #delta_phi_bins = round(delta_phi * nprofbin)
        #return int(delta_phi_bins)

    def fold_subs(self, input_array, period):
        """ Fold each band at given period"""

        folded=np.array([])

        # how many bins in one pulse period
        nbins_to_fold = int(np.rint(period/self.t_samp))
        # how many folds can we do with that number of bins
        nfolds = int(np.shape(input_array)[1]/nbins_to_fold)
        
        for band in input_array:
            # the array to fold into
            sub = np.zeros(nbins_to_fold)
            # do the fold
            for n in range(nfolds):
                indx = n*nbins_to_fold
                #print indx,nbins_to_fold
                sub = sub + band[indx:indx+nbins_to_fold]

            # add the fold to the array
            folded = np.append(folded,sub)

        folded = folded.reshape(np.shape(input_array)[0], -1)
        return folded

    def alt_bin_reduce(self, array, nbins=128):
        binned= np.zeros(nbins)
        bins_to_add = float(np.shape(array)[0])/nbins

        for n in range(nbins):
            indx1 = int(n*bins_to_add - bins_to_add/2)
            indx2 = int(n*bins_to_add+bins_to_add) - int(n*bins_to_add)
            
            # roll the array round so this starting bin is at zero
            # and take mean over the range bins_to_add
            binned[n] = np.mean(np.roll(array,-indx1)[:indx2])

        return binned

    def reduce_n_bins(self, array, nbins=128):
        binned= np.zeros(nbins)
    
        bins_to_add = float(np.shape(array)[0])/nbins

        for n in range(nbins):
            indx1 = int(n*bins_to_add)
            indx2 = int(n*bins_to_add+bins_to_add)
            try:
                binned[n] = np.mean(array[indx1:indx2])
            except IndexError:
                binned[n] = np.mean(array[indx1:])

        return binned

    def bin_2d_array(self, array, nbins=128):
        binned = np.array([])
        for band in array:
            binned = np.append(binned, self.reduce_n_bins(band, nbins))

        binned = binned.reshape(np.shape(array)[0], -1)
        return binned

    def normalise(self, array, normval=1.):
        return normval * array/np.max(array)
        
    def sumprof(self, profs):
        return profs.sum(0).sum(0)

    def avgprof(self, profs):
        return (profs/float(np.shape(profs)[2])).sum()

    def varprof(self,profs):
        var = 0.0
        for part in range(np.shape(profs)[0]):
            for sub in range(np.shape(profs)[1]):
                var += profs[part][sub].var()

        return var

    def calc_redchi2(self, profs):
        """ Calculate S Ransom's chi2 measure"""
        prof = self.sumprof(profs)
        avg = self.avgprof(profs)
        var = self.varprof(profs)
        #print prof, avg, var

        return ((prof-avg)**2.0/var).sum()/(len(prof)-1.0)


    def form_subs(self, input_array, nband=32, ax_num=0):
        """Chop an array up into bands, scrunch each band
            ax_num=0 is subbands
            ax_num=1 is subints
        """

        # how many bands do we want to sum?
        nchan = np.shape(input_array)[ax_num]
        if nchan%nband != 0:
            print "Number of bins is not integer multiple of number\
                    of sub-bins"
            sys.exit()
        else:
            nchan = nchan/nband

        # array to store the subbands/subints
        sub = np.array([])

        for n in range(nband):
            # cumulative sum of all channels in the band
            indx = n*nchan
            if ax_num == 0:
                sub = np.append(sub, input_array[indx:indx+nchan].sum(axis=0))
            else:
                sub = np.append(sub,input_array[:,indx:indx+nchan].sum(axis=1))

        sub = sub.reshape(nband, -1)
        return sub
