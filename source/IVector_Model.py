# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 13:59:49 2019

@author: Vo Thanh Phuong
"""

# Based on https://github.com/christianvazquez7/ivector/blob/master/MSRIT

from MFCCExtractor import mfcc_extractor
from GMModel import gm_model
import numpy as np

class IVectorModel:
    def __init__(self, n_components=1, covariance_type='full', tol = 1e-3, min_covar = 1e-3, max_iter=100, relevance_factor = 9, tv_dim = 400):
        self.n_componens = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.min_covar = min_covar
        self.max_iter = max_iter
        self.gmm_ubm = gm_model(self.n_componens, self.covariance_type, self.tol,
                                self.min_covar,
                                self.max_iter)
        self.relavance_factor = relevance_factor
        self.tv_dim = tv_dim
      
    def fit_ubm(self, data):
        self.gmm_ubm.fit(data)
        
    def compute_bw_stats(self, data):
        model = self.gmm_ubm
        detail = model.e_step(data)
        total = np.sum(detail, axis=0)
        F = []    
        
        for i in range(model.n_componens):
            cluster = model.clusters[i]
            mean = np.dot(detail[:,i].T, data) - (total[i] * cluster.mean)
            
            for item in mean:
                F.append(item)
        
        return total,np.array(F)
    
    def enroll(self, all_data):
        stats = []
        for idx in all_data:
            data = all_data[idx]            
            stats.append([self.compute_bw_stats(data)])
        
        