# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 23:44:51 2019

@author: Vo Thanh Phuong
"""

from MFCCExtractor import mfcc_extractor
from scipy.io import wavfile

fs, data = wavfile.read('../data/first.wav')
p = mfcc_extractor(data,fs)
print(p[0])