# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 00:37:12 2019

@author: Vo Thanh Phuong
"""
import numpy as np
import math
from scipy.fftpack import dct

def mel2hz(value):
    return 700 * (10**(value/2595.0) - 1)

def hz2mel(value):
    return 2595 * np.log10(1 + value / 700)

def nextpow2(value):
    return 1 << int((value-1)).bit_length()

def enframe(data, win, inc):
    result = []
    i = 0
    while True:
        if (i*inc+win < len(data)):
            result.append(data[i*inc:i*inc+win])
            i += 1;
        else:
            last = data[i*inc:len(data)]
            for j in range(win-len(last)):
                np.append(last, 0.0)
            result.append(last)
            break
    return result

def create_mel_filterbank(fs, nfft, nFilter, minHz, maxHz):
    filterbank = np.zeros((nFilter,nfft//2+1))
    minMel = hz2mel(minHz)
    maxMel = hz2mel(maxHz)
    
    Mel = np.linspace(minMel, maxMel, nFilter + 2)
    h = mel2hz(Mel)
    f = np.floor((nfft + 1) * h / fs)
    
    for m in range(nFilter):
        for k in range(int(f[m]), int(f[m+1])):
            filterbank[m, k] = (k - f[m]) / (f[m + 1] - f[m])
        for k in range(int(f[m + 1]), int(f[m + 2])):
            filterbank[m, k] = (f[m + 2] - k) / (f[m + 2] - f[m + 1])
 
    return filterbank

def lifter(input, L):
    if L <= 0:
        return input
    n_coef = len(input)
    output = np.zeros(n_coef)
    for i in range(n_coef):
         output[i] = input[i] * (1 + (L / 2) * np.sin(np.pi * i / L))
    return output
    
def mfcc_extractor(data, fs, win_ratio=0.025, overlap_ratio=0.015, num_filter=26, minHz=0, maxHz=8000):
    result = []
    
    win = round(fs * win_ratio)
    overlap = round(fs * overlap_ratio)
    
    data_frame = enframe(data, win, win - overlap)
    nfft = nextpow2(win)
    
    filterbank = create_mel_filterbank(fs, nfft, num_filter, minHz, maxHz)

    eps = math.exp(-30)
    for i in range(len(data_frame)):
        F = np.fft.rfft(data_frame[i], nfft)
        power_spectrum = (1.0/nfft) * np.square(np.absolute(F))       
        
        power_spectrum[power_spectrum < eps] = eps
        log_filterbank_energy = np.log(np.dot(filterbank, power_spectrum.T))
        tmp = dct(log_filterbank_energy, norm='ortho')       

        tmp = lifter(tmp[0:13], 22)
        
        energy = np.sum(power_spectrum)
        energy = np.where(energy == 0,np.finfo(float).eps,energy)
        tmp[0] = np.log(energy)
                
        result.append(tmp)
    
    return np.array(result)