# Import all libraries needed for the challenge
# General syntax to import specific functions in a library:
##from (library) import (specific library function)
# General syntax to import a library but no functions:
##import (library) as (give the library a nickname/alias)

from pandas import DataFrame, read_csv
from scipy.stats import norm
from scipy import optimize
from scipy import spatial
from numpy.random import normal
from numpy import arange, sin, pi
from sax import SAX

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
import pandas as pd #this is how I usually import pandas
import sys #only needed to determine Python version number
import numpy as np
import matplotlib.font_manager as fm
import os
import os.path
from matplotlib.pyplot import figure, show
import datetime,time
import scipy.io as sio

#import pylab as py
import matplotlib.pylab as pylab

RandomTS = np.fromfile('RandomTimeSeries.mat',dtype=float)
DataAndLabelsRandom = sio.loadmat('OwnTimeSeries.mat')['OwnTimeSeries']
RandomTSData = DataAndLabelsRandom[:,0]
RandomTSLabels = DataAndLabelsRandom[:,1]

activity = 'dense' #sparse, dense

if False:
    plt.plot(RandomTSData, label = "Random signal")
    plt.plot(RandomTSLabels, label = "Ground Truth", linewidth = 3)
    plt.legend()
    plt.xlabel('samples')
    plt.ylabel('units')
    plt.show()

matplotlib.colors.ColorConverter.colors['mc1'] = (0.976,0.333,0.518)

ECG_ref = sio.loadmat('ECG_ref.mat')['ECG_ref']
ECG_abn = sio.loadmat('ECG_abn.mat')['ECG_abn']

if False:
    plt.plot(ECG_ref, label = "Reference", color = 'g')
    plt.plot(ECG_abn, label = "Abnormal",color = 'mc1')
    plt.xlabel('samples')
    plt.ylabel('units')
    plt.legend()
    plt.show()

if activity == 'sparse':
    Data = sio.loadmat('ActivitySparse.mat')['ActivitySparse']
elif activity == 'dense':
    Data = sio.loadmat('ActivityDense.mat')['ActivityDense']

DataAndLabels = Data[0,0]
RealLife = DataAndLabels[:,0]

Labels = DataAndLabels[:,1]
SegmentList = Data[:,1] # or use RealLife[0,1]
SegList = SegmentList[0]

if False:
    plt.plot(RealLife, label="ReallifeData", color='b')
    plt.plot(Labels, label="Ground Truth", linewidth=3, color='g')

    plt.legend()
    plt.xlabel('samples')
    plt.ylabel('units')
    plt.show()

if False: ## Sine vs. random
    s = pd.Series(np.random.randn(100), name='something') # generate some random data.
    sx = SAX() # first an instance of the SAX object needs to be created.
    RandomSymbols = sx.to_letter_rep(s) # The output is denoted as RandomSymbols.

    t = arange(0.0, 1, 0.01)
    y = 5*sin(2*pi*t)
    sine_sx = SAX()
    Sine_Symbols = sine_sx.to_letter_rep(sin(2*pi*t))
    Segments = Sine_Symbols[1] # indicates where the time serise is segmente

    s = np.empty(len(Segments))
    seg = []
    for i in range(len(Segments)):
        helper = Segments[i]
        s = helper[0]
        seg.append(s)

    print("Time series length: ", len(y))
    print("Symbolic representation - Length of String: ", len(Sine_Symbols[0]))
    print("N of samples in in one segment:", len(Sine_Symbols[0])/float(len(y)))
    print("N of symbols to describe the segment:", len(Sine_Symbols[0])/float(len(y)))

    ECGref_sx = SAX()
    ECGabn_sx = SAX()
    (ECGref_Symbols, ECGref_Idx)= ECGref_sx.to_letter_rep(ECG_ref)
    (ECGabn_Symbols,ECGabn_Idx) = ECGabn_sx.to_letter_rep(ECG_abn)

    ref_abn_ComparisonScore = sx.compare_strings(ECGref_Symbols, ECGabn_Symbols) # two different signals compared
    print("The difference between the two strings is: ", ref_abn_ComparisonScore)

    ref_abn_ComparisonScore = sx.compare_strings(ECGref_Symbols, ECGref_Symbols) # the same signals compared
    print("The difference between the two strings is: ", ref_abn_ComparisonScore)

Real_sx = SAX()
Real_Symbols = Real_sx.to_letter_rep(RealLife)
Segments = Real_Symbols[1] # indicates where the time serise is segmente

s = np.empty(len(Segments))
seg = []
for i in range(len(Segments)):
    helper = Segments[i]
    s = helper[0]
    seg.append(s)
    #print(s)

print("Time series length: ", len(RealLife))
print("Symbolic representation - Length of String: ", len(Real_Symbols[0]))
print("N of samples in in one segment:", len(RealLife)/float(len(Real_Symbols[0])))
print("N of samples described by one symbol:", len(RealLife)/float(len(Real_Symbols[0])))

if True:
    plt.plot(RealLife, label="ReallifeData", color='b')
    plt.plot(Labels, label="Ground Truth", linewidth=3, color='g')
    plt.vlines(seg[:100], ymin=0, ymax=10, linewidth=2, color='m', label="Segments")
    plt.legend()
    plt.xlabel('samples')
    plt.ylabel('units')
    plt.show()


