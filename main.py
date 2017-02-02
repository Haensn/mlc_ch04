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


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

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

alphabetSizes = [3, 5, 10, 20]
wordSizes = [20, 50, 100]
thresholds = [3, 5, 7]
pathThreshs = [3, 4, 5]
numSubsequenceses = [100, 300, 500, 1000]
overlappingFraction = 0
levenshteinFractions = [0.2]



for alphabetSize in alphabetSizes:
    for wordSize in wordSizes:
        for threshold in thresholds:
            for pathThresh in pathThreshs:
                for numSubsequences in numSubsequenceses:
                    for levenshteinFraction in levenshteinFractions:

                        Real_sx = SAX(alphabetSize=alphabetSize, wordSize=wordSize)
                        #Real_Symbols = Real_sx.to_letter_rep(RealLife)
                        #Segments = Real_Symbols[1] # indicates where the time serise is segmente

                        windows = Real_sx.sliding_window(x=RealLife, numSubsequences=numSubsequences, overlappingFraction=overlappingFraction)

                        w, h = alphabetSize, alphabetSize
                        collision = [[0 for x in range(w)] for y in range(h)]

                        relevantWindows = []
                        motifClusters = []
                        discoveredClusters = [0]*len(RealLife)


                        for idx, window in enumerate(windows[0]):
                            previousLetter = None
                            collision = [[0 for x in range(w)] for y in range(h)]

                            #calculate transition frequencies
                            for letter in window:
                                if previousLetter != None:
                                    collision[ord(previousLetter) - 97][ord(letter) - 97] = collision[ord(previousLetter) - 97][ord(letter) - 97] + 1
                                previousLetter = letter

                            print("Current Window: ", window, " Position: ", windows[1][idx])
                            print("Current Matrix: ", collision)

                            #find path
                            pathIndices = []
                            for i in range(len(collision)):
                                for j in range(len(collision[i])):
                                    if collision[i][j] >= threshold:
                                        pathIndices.append((i,j))

                            pathFound = len(pathIndices) >= pathThresh

                            print(pathIndices)
                            print(pathFound)

                            if pathFound:
                                relevantWindows.append(window)

                                if len(motifClusters) == 0:
                                    motifClusters.append([window])
                                else:

                                    motifFound = False
                                    #check previous motifs
                                    for i, motifCluster in enumerate(motifClusters):
                                        distance = 0
                                        count = 0
                                        for motif in motifCluster:
                                            distance = distance + levenshtein(motif, window)
                                            count = count + 1
                                                 ##motifCluster.append(window)
                                        if distance/float(count) <= wordSize * levenshteinFraction:
                                            motifFound = True
                                            motifCluster.append(window)

                                            start = windows[1][idx][0]
                                            end = windows[1][idx][1]
                                            discoveredClusters[start:end] = [i+1] * (end - start)


                                            break
                                    if not motifFound:
                                        motifClusters.append([window])
                                        start = windows[1][idx][0]
                                        end = windows[1][idx][1]
                                        discoveredClusters[start:end] = [i + 1] * (end - start)





                        #s = np.empty(len(Segments))
                        #seg = []
                        #for i in range(len(Segments)):
                        #    helper = Segments[i]
                        #    s = helper[0]
                        #    seg.append(s)
                            #print(s)

                        #print("Time series length: ", len(RealLife))
                        #print("Symbolic representation - Length of String: ", len(Real_Symbols[0]))
                        #print("N of samples in in one segment:", len(RealLife)/float(len(Real_Symbols[0])))
                        #print("N of samples described by one symbol:", len(RealLife)/float(len(Real_Symbols[0])))

                        if True:
                            plt.plot(RealLife, label="ReallifeData", color='b')
                            plt.plot(Labels, label="Ground Truth", linewidth=3, color='g')
                            plt.plot(discoveredClusters, label="Discovered Clusters", linewidth=3, color='r')
                            #plt.vlines(seg[:100], ymin=0, ymax=10, linewidth=2, color='m', label="Segments")
                            plt.legend()
                            plt.xlabel('samples')
                            plt.ylabel('units')
                            filename = "al" + str(alphabetSize)+ "wo"+ str(wordSize)+ "th"+ str(threshold)+ "pa"+ str(pathThresh) + "nu"+ str(numSubsequences) + "le"+ str(levenshteinFraction) + ".png"
                            print(filename)
                            plt.savefig(filename)
                            plt.close()


