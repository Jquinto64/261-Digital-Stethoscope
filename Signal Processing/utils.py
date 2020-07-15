import csv
import numpy as np
import wave
import struct
from scipy import *
from pylab import *
from scipy.signal import find_peaks_cwt

# ignores all entries that are less than 2 seconds
def moreNsecs(file, N):
    f = wave.open(file)
    frames = f.readframes(-1)
    samples = struct.unpack('h'*f.getnframes(), frames)
    Fs = f.getframerate()
    t = [float(i)/Fs for i in range(len(samples))]
    if t[-1] > 2:
        return True
    else:
        return False
    
# Loads wav files 
def get_signal(file):
    f = wave.open(file)
    frames = f.readframes(-1)
    Fs = f.getframerate() # get sampling rate
    samples = struct.unpack('h'*f.getnframes(), frames)
    return np.array(samples), Fs # added sampling rate 

def find_peaks(samples, set_name):
    if set_name.upper() == 'A':
        interval = 200
        r = 5
    else:
        interval = 20
        r = 2
    all_peaks = []
    for sample in samples:
        indexes = find_peaks_cwt(sample, np.arange(1, r))
        peaks = []
        for i in indexes:
            if sample[i] > 0.13:
                peaks.append(i)

        if len(peaks) > 1:
            i = 1
            start = 0
            tmp_array = []
            max_peak = sample[peaks[start]]
            max_ind = start
            while i < len(peaks):
                if peaks[i] <= (peaks[start] + interval):
                    if sample[peaks[i]] > max_peak:
                        max_peak = sample[peaks[i]]
                        max_ind = i
                    if i == len(peaks)-1:
                        tmp_array.append(peaks[max_ind])
                        break
                    i += 1
                else:
                    tmp_array.append(peaks[max_ind])
                    start = i 
                    max_ind = start
                    max_peak = sample[peaks[start]]
                    i += 1
            peaks = tmp_array
        all_peaks.append(peaks)
    return np.array(all_peaks)

def get_S1S2_bounds(data, peaks, set_name):
    #finding difference between all peaks in every file
    all_diffs = []
    for k in range(len(peaks)):
        diff = np.diff(peaks[k])
        all_diffs.append(diff)
    
    #finding max difference or diastole period
    # and then labelling the first peak as s2 and second peak as s1
    max_index = []
    s1s2_peaks = []
    for k in range(len(all_diffs)):
        if any(all_diffs[k]):
            max_index.append(np.argmax(all_diffs[k]))
            s2 = peaks[k][max_index[k]]
            s1 = peaks[k][max_index[k]+1]
            s1s2_peaks.append([s1, s2])
        else:
            max_index.append(-1)
            s1s2_peaks.append([-1,-1])
    s1s2_peaks = np.array(s1s2_peaks)
    
    #defining s1 and s2 boundaries
    s1_bounds = []
    s2_bounds = []
    if set_name == 'A':
        upper_s1 = 200*2
        lower_s1 = 80*2
        upper_s2 = 600*2
        lower_s2 = 70*2
    else:
        upper_s1 = 25*10 
        lower_s1 = 10*10 
        upper_s2 = 35*10 
        lower_s2 = 10*10 
        
    for k in range(len(s1s2_peaks)):
        if s1s2_peaks[k][0] == -1:
            s1_bounds.append([-1,-1])
            s2_bounds.append([-1,-1])
        else:
            s1_lower = s1s2_peaks[k][0]-lower_s1
            s1_upper = s1s2_peaks[k][0]+upper_s1
            s2_lower = s1s2_peaks[k][1]-lower_s2
            s2_upper = s1s2_peaks[k][1]+upper_s2
            if s1_lower < 0:
                s1_lower = 0
            if s2_lower < 0:
                s2_lower = 0
            if s1_upper >= len(data[0]):
                s1_upper = len(data[0]) - 1
            if s2_upper >= len(data[0]):
                s2_upper = len(data[0]) - 1
            s1_bounds.append([s1_lower, s1_upper])
            s2_bounds.append([s2_lower, s2_upper])
        
    return np.array(s1_bounds), np.array(s2_bounds), s1_s2_peaks

#std deviation of specific interval where
#lower is the left most bound of the interval, upper is right most bound
def stdInterval(lower,low_index,upper,up_index, data):
    std = []
    for k in range(len(data)):
        if lower[k][0] == -1:
            std.append(0)
        else:
            dev = np.std(data[k][lower[k][low_index]:upper[k][up_index]])
            if np.isnan(dev):
                std.append(0)
            else:  
                std.append(dev)
    return np.array(std)


def freqInterval(data,lower,l_index,upper,u_index):
    freq = []
    for i in range(len(data)):
        if lower[i][0] == -1:
            freq.append(0)
        else:
            temp = data[i][lower[i][l_index]:upper[i][u_index]]
            temp = np.fft.fft(temp)
            temp = np.abs(temp)/max(np.abs(temp))
            #freq.append(temp[:int(len(temp)/2)])
            freq.append(temp)
    return np.array(freq)

def get_features(x_train,all_peaks):
    features = []
    for k in range(len(all_peaks)): 
        num_peaks = len(all_peaks[k])
        avg_between = 0
        avg_strength = 0
        if len(all_peaks[k]) == 1:
            avg_strength = x_train[k][all_peaks[k]]
        min_peak = 0
        max_peak = 0
        if num_peaks > 1:
            for i in range(1,len(all_peaks[k])):
                avg_between += np.abs(all_peaks[k][i] - all_peaks[k][i-1])
            avg_between /= len(all_peaks[k]) - 1
            min_peak = min(x_train[k][all_peaks[k]])
            max_peak = max(x_train[k][all_peaks[k]])
            for i in range(1,len(all_peaks[k])):
                avg_strength += x_train[k][all_peaks[k][i]]
            avg_strength /= len(all_peaks[k])
        features.append([num_peaks,avg_between,avg_strength,min_peak,max_peak])
    features = np.array(features)
    maxes = np.amax(features,axis=0)
    for feature in features:
        feature = np.array([feature[0]/maxes[0],feature[1]/maxes[1],feature[2]/maxes[2],feature[3]/maxes[3],feature[4]/maxes[4]])
    return features 

