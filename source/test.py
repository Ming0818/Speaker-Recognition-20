# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 23:44:51 2019

@author: Vo Thanh Phuong
"""

from MFCCExtractor import mfcc_extractor
from GMModel import gm_model
from GM_UBModel import gmmubm_model
from scipy.io import wavfile
from sklearn.mixture import GaussianMixture
import numpy as np
import pickle
import os

gm_models = []
gm_models_sklean = []
label = [ '1', '2', '3', '4', '5' ]

train_words = [ 'coffee', 'hello', 'laptop', 'mobile', 'music', 'speech' ]
test_words = train_words

ubm_data = []
all_data = {}

for i in label:    
    data = []
    for j in train_words:
        url = '../data/train/{}/{}.wav'.format(i,j)
        fs,x = wavfile.read(url)
        f = mfcc_extractor(x, fs, maxHz = fs / 2)
        
        if (len(data) == 0):
            data = f
        else:
            data = np.concatenate((data,f), axis = 0)
    
        if (len(ubm_data) == 0):
            ubm_data = f
        else:
            ubm_data = np.concatenate((ubm_data, f), axis = 0)

    all_data.update({i:data})

    # build background model
ubm_model_direct = 'ubmodel.mat'
if os.path.isfile(ubm_model_direct):
    filehandler = open(ubm_model_direct, 'rb')
    ub_gm_model = pickle.load(filehandler)
    filehandler.close()
else:
    ub_gm_model = gmmubm_model(n_components=6, max_iter=1000)
    ub_gm_model.fit_ubm(ubm_data)   
    filehandler = open(ubm_model_direct, 'wb')
    pickle.dump(ub_gm_model, filehandler)
    filehandler.close()

    # adapt MAP    
for i in all_data:
    ub_gm_model.fit_gmm(i, all_data[i])
    
num_test = len(label) * len(test_words)
num_true = 0

    # test
for i in label:    
    for j in test_words:
        url = '../data/train/{}/{}.wav'.format(i,j)
        fs,x = wavfile.read(url)
        f = mfcc_extractor(x, fs, maxHz = fs / 2)
        
        if (ub_gm_model.predict(f) == i):
            num_true += 1
            
print('Accuracy = {}'.format(num_true/num_test))