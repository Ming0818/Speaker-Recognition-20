# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 00:37:12 2019

@author: Vo Thanh Phuong
"""
import numpy as np

def mel2hz(value):
    return 700 * (10**(value/2595) - 1)

def hz2mel(value):
    return 2595 * np.log10(1 + value / 700)

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
                last.append(0)
            result.append(last)
            break
    return result

def createMelFilterbank(fs, nfft, nFilter, minHz, maxHz):
    filterbank = np.zeros((nFilter,nfft))
    minMel = hz2mel(minHz)
    maxMel = hz2mel(maxHz)
    
    Mel = np.linspace(minMel, maxMel, nFilter + 2)
    h = mel2hz(Mel)
    f = np.floor((nfft*2-1)*h/fs)
    
    print(f)
    
createMelFilterbank(16000, 257, 26, 0, 8000)