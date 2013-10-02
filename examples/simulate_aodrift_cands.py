#!/usr/bin/python

import os
import sys
import math
import random
import cPickle

from pfdsim import Pulsar
from pfdsim import Survey
from pfdsim import SimPfd

def calc_num_bins(period):
    if period < 0.002:
        return 24,64
    elif period < 0.05:
        return 50,32
    elif period<0.5:
        return 100,16
    else:
        return 200,8

def drawlnorm(mean, sigma):
    return 10.0**random.gauss(mean, sigma)

def draw_lorimer2012_period():
    logpmin = 0.
    logpmax = 1.5
    dist = [1.,3.,5.,16.,9.,5.,5.,3.,2.]

    bin_num = draw1d(dist)

    logp = logpmin + (logpmax-logpmin)*(bin_num*random.random())/len(dist)

    return 10.**logp

def draw1d(dist):
    total = sum(dist)
    cumulative = [sum(dist[:x+1])/total for x in range(len(dist))]

    rand_num = random.random()
    for i, c in enumerate(cumulative):
        if rand_num<=c:
            return i

def main(sourcetype, outdir):
    # pick period from a log-normal distribution, mean = 2.7
    # stddev = -0.34
    # or dunc's msp distribution for msps
    if sourcetype == 'pulsar':
        period = drawlnorm(2.7, -.34)/1000.

    elif sourcetype == 'msp':
        period = draw_lorimer2012_period()/1000.

    # for fake pulsars
    dm = random.uniform(4.,500.)
    # for dm=0 RFI
    #dm = random.uniform(-2.,2.)
    snr = random.expovariate(.16)
    # for narrow
    #duty = random.gauss(0.06,.03)
    # for wide
    duty = random.gauss(0.3,.2)
    if duty<0.:
        duty = 0.-duty

    smearing=True

    print "creating pulsar"
    pulsar = Pulsar(period, dm, snr, duty, smearing)

    print "creating survey"
    # define aodrift survey parameters
    # might need to check on these
    n_profile_bins, n_subints = calc_num_bins(pulsar.period)
    aodrift = Survey(n_bits=2,
                     t_samp = 256.E-6,
                     n_samp = 262144,
                     n_freq_chans = 512,
                     freq_chan_width_mhz = 0.0488,
                     freq_centre = 327.,
                     n_sub_bands=32,
                     n_sub_ints=n_subints,
                     n_prof_bins=n_profile_bins)

    print "creating pfd"
    # create raw data
    simpfd = SimPfd(pulsar, aodrift)
    filestring = '{0}_{1:.1f}_{2:.1f}_{3:.2f}_{4:.4f}.simpfd'.format(sourcetype,
                                                     dm,
                                                     snr,
                                                     duty,
                                                     period)

    resultpath = outdir
    outf = os.path.join(resultpath, filestring)

    # write the pfd to a file
    f = open(outf, 'wb')
    cPickle.dump(simpfd, f)
    f.close()

if __name__ == '__main__':
    try:
        sourcetype = sys.argv[1]
        outdir = sys.argv[2]
    except IndexError:
        sourcetype='pulsar'
        outdir = '/home/sbates/pfdsim/fake_pulsars'

    # run script
    main(sourcetype, outdir)
