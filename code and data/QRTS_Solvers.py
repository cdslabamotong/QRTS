# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:27:47 2021

@author: Amo
"""
import numpy as np
import sys
import math
import random
#import time
import copy
import heapq
import multiprocessing
from Utils import Utils, Dijkstra
from shutil import copyfile
from USCO import USCO
from scipy.stats import powerlaw
from StoGraph import StoGraph, Graph
#from base import StructuredModel
from USCO_Solver import USCO_Solver
from one_slack_ssvm_normal import OneSlackSSVM as OneSlackSSVM_normal
import time
class QRTS_P(USCO_Solver):

    def __init__(self):
        pass

    def initialize(self, realizations, USCO):
        # set any data-specific parameters in the model
        # self.featureNum = instance.featureNum
        self.P_realizations = realizations
        self.P_realizationNum = len(realizations)
        self.inference_calls = 0
        self.USCO = USCO

    def train(self, trainMethod, verbose, C, tol, thread, max_iter, logpath, TrainQueries, TrainDecisions, beta):
        if trainMethod == "one_slack_normal":
            self.realizations = self.P_realizations
            self.size_joint_feature = self.P_realizationNum
            one_slack_svm = OneSlackSSVM_normal(self, verbose=verbose, C=C, tol=tol, n_jobs=thread,
                                                max_iter=max_iter, log=logpath)
            one_slack_svm.fit(TrainQueries, TrainDecisions, beta, initialize=False)
            return one_slack_svm.w
        pass

class QRTS_D(USCO_Solver):

    def __init__(self):
        pass

    def initialize(self, D_realizationNum, USCO, family = "exp"):
        # set any data-specific parameters in the model
        # self.featureNum = instance.featureNum
        #self.realizations = USCO.genSimpleEdgeRealizations()
        self.D_realizationNum = D_realizationNum
        #self.ifStopII = ifStopII
        self.inference_calls = 0
        self.USCO = USCO
        self.family = family
        self.current_d_em = copy.deepcopy(self.USCO.stoGraph.nodes)

        if self.family == "exp":
            for node1 in self.current_d_em:
                for node2 in self.current_d_em[node1].neighbor:
                    alpha = random.random() + 0.1
                    lamb = random.random() * 10
                    mean = Utils.getWeibull_alpha_lambda(alpha, lamb)
                    #std = 1
                    self.current_d_em[node1].neighbor[node2] = mean
        #self.D_realizations = self.P1_sampling(5)
        #self.D_realizations = self.P1_sampling(5)
        # time.sleep(3)
        # multiprocessing.set_start_method("spawn",force=True)
        # self.D_realizations = self.P1_sampling_2(10, p)

    def train(self, trainMethod, verbose, C, tol, thread, max_iter_inner, logpath, TrainQueries, TrainDecisions, beta, max_iter):
        #if max_iter > 0 and self.ifTune is False:
            #sys.exit("max_iter > 0 and self.ifTune is False")

        for i in range(max_iter):
            print("QRTS_D {} *************************************************".format(i))
            self.D_realizations = self.P1_sampling(thread)
            print("P1_sampling Done------------------------------------------".format(i))
            self.w = self.P2_importance_learning(trainMethod, verbose, C, tol, thread, max_iter_inner, logpath, TrainQueries, TrainDecisions, beta)
            print("P2_importance_learning Done------------------------------------------".format(i))

            self.P3_distribution_tuning()
            print("P3_distribution_tuning Done------------------------------------------".format(i))


        print("Final Distribution learning------------------------------------------")
        self.D_realizations = self.P1_sampling(thread)
        self.w = self.P2_importance_learning(trainMethod, verbose, C, tol, thread, max_iter_inner, logpath, TrainQueries,
                                             TrainDecisions, beta)

        return self.w, self.D_realizations
        pass

    def P1_sampling(self,thread):
        realizations = []
        thread = 1
        if thread == 1:
            for cout in range(self.D_realizationNum):
                realization = self.P1_sampling_one(self.USCO.stoGraph.vNum, 0)
                realizations.append(realization)
            #print(cout)
        else:
            print("P1_sampling {}".format(thread))
            p = multiprocessing.Pool(thread)
            # print("222")
            realizations=p.starmap(self.P1_sampling_one, [(self.USCO.stoGraph.vNum, i) for i in
                                                 range(self.D_realizationNum)])
            # print("333")
            p.close()
            #p.terminate()
            p.join()
        return realizations


    def P1_sampling_one(self, vNum, nonsense):
        #time.sleep(5)
        sample_current_d_em = copy.deepcopy(self.current_d_em)
        for node1 in sample_current_d_em:
            for node2 in sample_current_d_em[node1].neighbor:
                if self.family == "exp":
                    mean  = self.current_d_em[node1].neighbor[node2]
                    #std = self.current_d_em[node1].neighbor[node2][1]
                    if mean<0:
                        mean = 0.01
                    weight = np.random.exponential(mean, 1)[0]
                    #print(weight)
                    sample_current_d_em[node1].neighbor[node2] = weight
        realization = self.USCO.Realization()
        realization.initialize_matrix(sample_current_d_em, vNum)

        self.distance = {}
        return realization


    def P2_importance_learning(self,trainMethod, verbose, C, tol, thread, max_iter, logpath, TrainQueries, TrainDecisions, beta):
        if trainMethod == "one_slack_normal":
            self.realizations = self.D_realizations
            self.size_joint_feature = self.D_realizationNum
            one_slack_svm = OneSlackSSVM_normal(self, verbose=verbose, C=C, tol=tol, n_jobs=thread,
                                                max_iter=max_iter, log=logpath)
            one_slack_svm.fit(TrainQueries, TrainDecisions, beta, initialize=False)
            return one_slack_svm.w



    def P3_distribution_tuning(self):
        for node1 in self.current_d_em:
            for node2 in self.current_d_em[node1].neighbor:
                if self.family == "exp":
                    mean  = self.current_d_em[node1].neighbor[node2]
                    #std = self.current_d_em[node1].neighbor[node2][1]
                    mean = 0

                    for i in range(self.D_realizationNum):
                        mean += self.w[i]*self.D_realizations[i].weightMatrix[node1][node2]
                    self.current_d_em[node1].neighbor[node2] = mean
                    #weight = np.random.exponential(mean, 1)[0]
                    #print(weight)
                    #sample_current_d_em[node1].neighbor[node2] = weight



class QRTS_PD(QRTS_D):

    def __init__(self):
        pass

    def initialize(self, D_realizationNum, P_realizaitons, USCO, family = "exp"):
        # set any data-specific parameters in the model
        # self.featureNum = instance.featureNum
        #self.realizations = USCO.genSimpleEdgeRealizations()

        self.D_realizationNum = D_realizationNum
        self.P_realizations = P_realizaitons
        self.P_realizationNum = len(P_realizaitons)
        self.inference_calls = 0
        self.USCO = USCO
        self.family = family
        self.current_d_em = copy.deepcopy(self.USCO.stoGraph.nodes)

        # if self.family == "exp":
        #     self.current_d_em = copy.deepcopy(self.USCO.stoGraph.nodes)
        #     for node1 in self.current_d_em:
        #         for node2 in self.current_d_em[node1].neighbor:
        #             alpha = random.random() + 0.1
        #             lamb = random.random() * 10
        #             mean = Utils.getWeibull_alpha_lambda(alpha, lamb)
        #             #std = 1
        #             self.current_d_em[node1].neighbor[node2] = mean




    def train(self, trainMethod, verbose, C, tol, thread, max_iter_inner, logpath, TrainQueries, TrainDecisions, beta, iterNum):
        self.P0_point_initialize(trainMethod, verbose, C, tol, thread, max_iter_inner, logpath, TrainQueries, TrainDecisions, beta)

        # self.current_d_em = copy.deepcopy(self.USCO.stoGraph.nodes)
        # for node1 in self.current_d_em:
        #     for node2 in self.current_d_em[node1].neighbor:
        #         alpha = random.random() + 0.1
        #         lamb = random.random() * 10
        #         mean = Utils.getWeibull_alpha_lambda(alpha, lamb)
        #         #std = 1
        #         self.current_d_em[node1].neighbor[node2] = mean

        for i in range(iterNum):
            print("QRTS_D {}------------------------------------------".format(i))
            self.D_realizations = self.P1_sampling(thread)
            print("P1_sampling Done------------------------------------------".format(i))
            self.w = self.P2_importance_learning(trainMethod, verbose, C, tol, thread, max_iter_inner, logpath, TrainQueries, TrainDecisions, beta)
            print("P2_importance_learning Done------------------------------------------".format(i))
            self.P3_distribution_tuning()
            print("P3_distribution_tuning Done------------------------------------------".format(i))

        print("Final Distribution learning------------------------------------------")
        self.D_realizations = self.P1_sampling(thread)
        self.w = self.P2_importance_learning(trainMethod, verbose, C, tol, thread, max_iter_inner, logpath, TrainQueries,
                                             TrainDecisions, beta)
        return self.w, self.D_realizations


    def P0_point_initialize(self, trainMethod, verbose, C, tol, thread, max_iter, logpath, TrainQueries, TrainDecisions, beta):
        print("P_initialize------------------------------------------")
        if trainMethod == "one_slack_normal":
            self.realizations = self.P_realizations
            self.size_joint_feature = self.P_realizationNum
            one_slack_svm = OneSlackSSVM_normal(self, verbose=verbose, C=C, tol=tol, n_jobs=thread,
                                                max_iter=max_iter, log=logpath)
            one_slack_svm.fit(TrainQueries, TrainDecisions, beta, initialize=False)

        for i in range(self.P_realizationNum):
            cout = 0
            for node1 in self.P_realizations[i].weightMatrix:
                for node2 in self.P_realizations[i].weightMatrix[node1]:
                    if cout > 0:
                        sys.exit("cout > 0")
                    self.current_d_em[node1].neighbor[node2] = one_slack_svm.w[i]
                    cout += 1



class QRTS_D_plus(QRTS_D):

    def __init__(self):
        pass

    def initialize(self, D_realizationNum, USCO, family = "exp"):
        # set any data-specific parameters in the model
        # self.featureNum = instance.featureNum
        #self.realizations = USCO.genSimpleEdgeRealizations()

        self.D_realizationNum = D_realizationNum
        #self.P_realizations = P_realizaitons
        #self.P_realizationNum = len(P_realizaitons)
        self.inference_calls = 0
        self.USCO = USCO
        self.family = family
        self.current_d_em = copy.deepcopy(self.USCO.stoGraph.nodes)

        if self.family == "exp":
            #self.current_d_em = copy.deepcopy(self.USCO.stoGraph.nodes)
            for node1 in self.current_d_em:
                for node2 in self.current_d_em[node1].neighbor:
                    alpha = random.random() + 0.1
                    lamb = random.random() * 10
                    mean = Utils.getWeibull_alpha_lambda(alpha, lamb)
                    #std = 1
                    self.current_d_em[node1].neighbor[node2] = mean

    def train(self, trainMethod, verbose, C, tol, thread, max_iter_inner, logpath, TrainQueries, TrainDecisions, beta,
              max_iter):
        # if max_iter > 0 and self.ifTune is False:
        # sys.exit("max_iter > 0 and self.ifTune is False")
        self.D_realizations = self.P0_sampling(thread)
        print("P0_sampling Done------------------------------------------")
        self.w = self.P2_importance_learning(trainMethod, verbose, C, tol, thread, max_iter_inner, logpath,
                                             TrainQueries, TrainDecisions, beta)
        print("P0_importance_learning Done------------------------------------------")

        for i in range(max_iter):
            print("QRTS_D {} *************************************************".format(i))
            self.D_realizations = self.P1_sampling(thread)
            print("P1_sampling Done {}------------------------------------------".format(i))
            self.w = self.P2_importance_learning(trainMethod, verbose, C, tol, thread, max_iter_inner, logpath,
                                                 TrainQueries, TrainDecisions, beta)
            print("P2_importance_learning Done {}------------------------------------------".format(i))

            self.P3_distribution_tuning()
            print("P3_distribution_tuning Done {}------------------------------------------".format(i))

        print("Final Distribution learning------------------------------------------")
        self.D_realizations = self.P1_sampling(thread)
        self.w = self.P2_importance_learning(trainMethod, verbose, C, tol, thread, max_iter_inner, logpath,
                                             TrainQueries,
                                             TrainDecisions, beta)

        return self.w, self.D_realizations

    def P0_sampling(self, thread):
        realizations = []
        thread = 1
        if thread == 1:
            for cout in range(self.D_realizationNum):
                realization = self.P0_sampling_one(self.USCO.stoGraph.vNum, 0)
                realizations.append(realization)
            # print(cout)
        else:
            print("P1_sampling {}".format(thread))
            p = multiprocessing.Pool(thread)
            # print("222")
            realizations = p.starmap(self.P0_sampling_one, [(self.USCO.stoGraph.vNum, i) for i in
                                                            range(self.D_realizationNum)])
            # print("333")
            p.close()
            # p.terminate()
            p.join()
        return realizations

    def P0_sampling_one(self, vNum, nonsense):
        # time.sleep(5)
        sample_current_d_em = copy.deepcopy(self.current_d_em)
        for node1 in sample_current_d_em:
            for node2 in sample_current_d_em[node1].neighbor:
                alpha = random.random() + 0.1
                lamb = random.random() * 10
                mean = Utils.getWeibull_alpha_lambda(alpha, lamb)
                sample_current_d_em[node1].neighbor[node2] = mean
        realization = self.USCO.Realization()
        realization.initialize_matrix(sample_current_d_em, vNum)

        self.distance = {}
        return realization





def main():
    pass

    
    
def temp():
    pass

#g = Graph(9)
if __name__ == "__main__":
    pass
    main()
    