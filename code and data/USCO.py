# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:27:47 2021
USCO_Solver: using Structured SVM for solving UCCO problem; needs implementation for each instance
USCO: an instance of USCO problem; needs implementation for each problem

@author: Amo
"""
import numpy as np
import sys
import math
import random
import copy
import multiprocessing
#import Utils
sys.path.insert(0,'..')
from Utils import Dijkstra, Utils


class USCO(object):
    
    '''
    class Pair(object):
        def __init__(self, x, y):
            #self.index = index
            self.x = x
            self.y = y
            #self.out_degree = 0    
    '''
    '''
    stoGraph: a stograph
    obeValue
    def test(self, X_test, Y_length, Y_pred, logpath= None)
    def genPairs(path, stoGraph, num):
    '''
    class Realization(object):
        def __init__(self):
            raise NotImplementedError()
            
    class Query(object):
        def __init__(self):
            
            raise NotImplementedError()
            
    class Sample(object):
        def __init__(self):
            
            raise NotImplementedError()
    
    def __init__(self, stoGraph, useModel = False):
        self.stoGraph = stoGraph
        self.useModel = useModel
        #self.realizations, self.realizationIndexes = self.readRealizations(realizationPath, realizationNum)
        #self.realizations = None
        #self.realizationIndexes = None
        #self.pairs = None
        #self.samples = self.readSamples(samplePath, sampleNum)
        #self.samples = None
        #self.vNum=vNum
        
    #def readRealizations(self, realizationPath, realizationNum)
    
    def kernel(self, realization, query, y):
        raise NotImplementedError()
        
        
        
    def objValue(self, realizations, W, x, y):
        value = 0
        for realization, w, in zip(realizations, W):
            r_value= self.kernel(realization,x,y)
            value = value + w*r_value
        
        return value 
    
    def computeScore(self, x, y, w):
        feature = self.computeFeature(x, y)
        return w.dot(feature)
        
    def computeFeature(self, realizations, x, y):
        feature = []
        #print(self.featureNum)
        for realization in realizations:
            #feature.append(1)
            feature.append(self.kernel(realization, x, y))
        #print(feature)
        return feature

    def computeFeature_batch(self, realizations, X, Y, n_jobs=1):
        joint_feature_ = np.zeros(len(realizations))
        if len(X) > 2000:
            chunksize = 300
        else:
            chunksize = 30
        p = multiprocessing.Pool(n_jobs)
        #print(len(X))
        #print(len(Y))
        Ys = p.starmap(self.computeFeature, [(realizations, X[i], Y[i]) for i in range(len(X))], chunksize = chunksize)
        p.close()
        p.join()

        for feature in Ys:
            joint_feature_ += np.array(feature)

        return joint_feature_
        #raise NotImplementedError()

    def solve_R(self, realizations, W, x):
        '''
        Define how to compute y = max w^T[f(x,y,r_1),...f(x,y,r_n)]
        '''
        raise NotImplementedError()
    
    def solve_R_batch(self, X, W, realizations, n_jobs=1, offset = None, trace = True):
        #print("inference") 
        print("solve_R_batch RUNNING")     
        
        if n_jobs == 1:
            result =[]
            for x in X:
                result.append(self.solve_R(realizations, W, x))
            print("solve_R_batch DONE")
            return result
        else:
            print("111")
            results={}
            p = multiprocessing.Pool(n_jobs)
            #print("222")
            results=p.starmap(self.solve_R, ((realizations, W, x) for x in X))
            p.close()
            p.join()
            print("solve_R_batch DONE")
            return results  
        
    def solveTrue(self, x):
        '''
        Define how to compute the true solution
        '''
        raise NotImplementedError()
    '''  
    def solveBatch(self, realizations, W, X, thread = 1):
        
        Define how to compute y = max w^T[f(x,y,r_1),...f(x,y,r_n)]
        
        raise NotImplementedError()
    '''
    def genQuery(self):
        raise NotImplementedError()
        
            
       
    def genSamples(self, Wpath, trainNum, testNum, Max = None):
        raise NotImplementedError()
    
    def readSamples(self, Rpath, trainNum, testNum, Max = None):
        raise NotImplementedError()
        
    
    def readRealizations(self, Rpath, fetureNum, Max = None):
        raise NotImplementedError()
    
    def test(self, samples, PredDecisions):
        raise NotImplementedError()
    
    



if __name__ == "__main__":
    print(np.random.exponential(1,1)[0])
    pass

    #stoGraph=StoGraph("data/pl/pl_model", 768)
    #stoGraph.GenStoGraph("data/pl/pl_model")
    #stoGraph.genMultiRealization_P(10000, "data/pl/features/true_10000/", edgeProb= 0.1, weightType="true", thread=5, startIndex= 0, distance = True)
