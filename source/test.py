# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 23:44:51 2019

@author: Vo Thanh Phuong
"""

from MFCCExtractor import mfcc_extractor
from GMModel import gm_model
from scipy.io import wavfile
from sklearn.mixture import GaussianMixture
import numpy as np

gm_models = []
gm_models_sklean = []
label = [ '1', '2', '3', '4', '5' ]

train_words = [ 'coffee', 'hello', 'laptop', 'mobile', 'music' ]
test_words = [ 'speech' ]

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
    
    gmm = gm_model(n_components=6)
    gmm.fit(data)
    gm_models.append(gmm)

    gmm = GaussianMixture(n_components=6, random_state=999)
    gmm.fit(data)
    gm_models_sklean.append(gmm)


num_case = len(label) * len(test_words)
num_true = 0
num_true_sklearn = 0

for i in range(len(label)):    
    data = []
    for j in test_words:
        url = '../data/train/{}/{}.wav'.format(label[i],j)
        fs,x = wavfile.read(url)
        f = mfcc_extractor(x, fs, maxHz = fs / 2)
        
        max_score = -np.Inf
        idx = -1
        
        max_score_1 = -np.Inf
        idx1 = -1
        for ii in range(len(gm_models)):
            if (gm_models[ii].score(f) > max_score):
                max_score = gm_models[ii].score(f)
                idx = ii                
            
            if (gm_models_sklean[ii].score(f) > max_score_1):
                max_score_1 = gm_models_sklean[ii].score(f)
                idx1 = ii       
        
        print(max_score)
        print(max_score_1)
        
        if (idx == i):
            num_true += 1
            
        if (idx1 == i):
            num_true_sklearn += 1
            

print('model: {}'.format(num_true/num_case))
print('sklearn model: {}'.format(num_true_sklearn/num_case))