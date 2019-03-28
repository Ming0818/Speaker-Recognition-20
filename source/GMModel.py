# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 00:41:57 2019

@author: Vo Thanh Phuong
"""

from MFCCExtractor import mfcc_extractor
from sklearn.cluster import KMeans
import numpy as np

class gm_cluster:
    def __init__(self, prior, data, min_covar):
        self.prior = prior
        self.mean = np.mean(data, axis=0)
        self.covariance = np.cov(data, rowvar=0) + min_covar * np.eye(len(self.mean))
        self.dim = len(self.mean)
        self.factor = 1.0 / (((2 * np.pi) ** (self.dim / 2)) * (np.linalg.det(self.covariance) ** 0.5))

    def pdf(self, x):
        tmp = x - self.mean
        inv_covar = np.linalg.inv(self.covariance)
        return self.factor * np.exp(-0.5 * np.dot(np.dot(tmp,inv_covar), tmp.T))

class gm_model:
    def __init__(self, n_components=1, covariance_type='full', tol = 1e-3, min_covar = 1e-3, max_iter=100):
        self.n_componens = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.min_covar = min_covar
        self.max_iter = max_iter
   
    def gmm_init_kmeans(self, data):
        label = (KMeans(self.n_componens)).fit(data).labels_
        clusters = []
        for i in range(self.n_componens):
            sub_data = data[label == i]
            clusters.append(gm_cluster(len(sub_data)/len(label), sub_data, 
                                       self.min_covar))
        return clusters
    
    def e_step(self):
        return
    
    def m_step(self):
        return
    
    def fit(self, data):
        clusters = self.gmm_init_kmeans(data)
        log_likelihood = []
        for step in range(self.max_iter):
            self.e_step()
            self.m_step()
        return
    
    def score(self, data):
        return
    
model = gm_model(n_components=2)
model.fit(np.array([[1,2,3],[1,2,3], [1.1, 2.1, 3.1], [3,4,5], [3,4,5]]))