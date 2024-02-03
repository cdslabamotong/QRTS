"""
==============================
Social Inverse Training
==============================
"""
import os
import sys
import multiprocessing
import argparse
import numpy as np
from datetime import datetime


from one_slack_ssvm_normal import OneSlackSSVM as OneSlackSSVM_normal
from one_slack_ssvm_lowerBound import OneSlackSSVM as OneSlackSSVM_lowerBound
from subgradient_ssvm import SubgradientSSVM
from n_slack_ssvm import NSlackSSVM


sys.path.insert(0,'..')
from DE import DE_USCO
from DC import DC_USCO
from SP import SP_USCO
from Steiner import Steiner_USCO
from StoGraph import StoGraph
from USCO_Solver import USCO_Solver
from Utils import Utils
from QRTS_Solvers import QRTS_P



class Object(object):
    pass


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--sourceT',  default='Steiner',
                        choices=['DE','DC','SP','Steiner'])
    
    parser.add_argument(
        '--targetT',  default='Steiner',
                        choices=['DE','DC','SP','Steiner'])
    
    
    parser.add_argument(
        '--graph',  default='col_3',
                        choices=['col_3','kro_2','ws','ws_sparse','ba'])
    
    parser.add_argument(
        '--graphType',  default='Weibull',
                        choices=['Gaussian_1_10_100','Weibull'])
    #parser.add_argument(
    #    '--vNum', type=int, default=768, choices=[1024,768,512],
    #                    help='kro 1024, power768 768, ER512 512')
    
    
    parser.add_argument(
        '--featureNum', type=int, default=10,
                        help='number of features (random subgraphs) used in StratLearn ')
    #parser.add_argument(
    #    '--featureGenMethod', default='exp',
    #         choices=['uniform','exp','true','edge_unit','em_approx_10_10'],
    #            help='the distribution used for generating features, the choices correspond phi_1^1, phi_0.01^1, phi_0.005^1, phi_+^+')
    
        
    parser.add_argument(
        '--trainNum', type=int, default=1000, help='number of training data')
    
    parser.add_argument(
        '--testNum', type=int, default=10, help='number of testing data')
    
    parser.add_argument(
        '--testBatch', type=int, default=5, help='number of testing data')   
     
    parser.add_argument(
        '--thread', type=int, default=5, help='number of threads')
    
    parser.add_argument(
        '--beta', type=float, default=1, help='number of threads')
    
    parser.add_argument(
        '--output', default=False, action="store_true", help='if output prediction')
    
    
    parser.add_argument(
        '--pre_train', default=True ,action="store_true", help='if store a pre_train model')
    
    parser.add_argument(
        '--log_path', default='log',  help='if store a pre_train model')

    parser.add_argument(
        '--max_iter', type=int, default=1, help='if store a pre_train model')
    
    args = parser.parse_args()
    #utils= Utils.Utils
    
    #problem ="ssp"
    
    sourceT = args.sourceT
    targetT = args.targetT
    
    
    graph=args.graph
    graphType=args.graphType
    #vNum = args.vNum
    
    
    
    trainNum =args.trainNum
    testNum =args.testNum
    testBatch =args.testBatch
    #pairMax=2500
    
    thread = args.thread
    
    
    verbose=3
    #parameter used in SVM
    C = 0.0001
    tol=0.001
    max_iter = args.max_iter
    #alpha = 0.618
    #beta = alpha*alpha*(1-args.beta)
    beta = args.beta
    featureNum = args.featureNum
    featureGenMethod = "edge_unit"
    sourceSample_maxPair = 10000
    targetSample_maxPair = 1000
    sourceSampleFileName = "{}_{}_{}_samples".format(graph, graphType, sourceT)
    targetSampleFileName = "{}_{}_{}_samples".format(graph, graphType, targetT)

    if graph == "kro_2":
        vNum = 1024
        eNum = 2745


    if graph == "col_3":
        vNum = 512
        eNum = 1551

    if graph == "ws":
        vNum = 1001
        eNum = 10000

    if graph == "ws_sparse":
        vNum = 1001
        eNum = 5000

    if graph == "ba":
        vNum = 501
        eNum = 1500

    if featureGenMethod == 'edge_unit':
        maxFeatureNum = eNum
        featureNum = maxFeatureNum
    else:
        maxFeatureNum = 10000




    
    pre_train = args.pre_train
    preTrainPathResult = None

    
    #get data
    path = os.getcwd() 
    data_path=os.path.dirname(path)+"/data"
    sourceSample_path = "{}/{}/{}".format(data_path, graph, sourceSampleFileName)
    targetSample_path = "{}/{}/{}".format(data_path, graph, targetSampleFileName)
    stoGraphPath = "{}/{}/{}_{}_stoGraph".format(data_path, graph, graph, graphType)
    featurePath = "{}/{}/features/{}_{}".format(data_path, graph, featureGenMethod, maxFeatureNum)
    if args.log_path is not None:
        logpath=path+"/"+args.log_path

    stoGraph = StoGraph(stoGraphPath, vNum, graphType)

    


    if sourceT == "SP":
        source_USCO = SP_USCO(stoGraph)
    if sourceT == "Steiner":
        source_USCO = Steiner_USCO(stoGraph, useModel = True)
    

    if targetT== "SP":
        target_USCO = SP_USCO(stoGraph)
    if targetT== "Steiner":
        target_USCO = Steiner_USCO(stoGraph, useModel = True)
        
    
    #
    
    TrainSamples, TrainQueries, TrainDecisions, = source_USCO.readSamples(sourceSample_path, trainNum, sourceSample_maxPair)
    TestSamples_s, TestQueries_s, TestDecisions_s = [], [], []
    for i in range(testBatch):
        TestSamples, TestQueries, TestDecisions = target_USCO.readSamples(targetSample_path, testNum, targetSample_maxPair)
        TestSamples_s.append(TestSamples)
        TestQueries_s.append(TestQueries)
        TestDecisions_s.append(TestDecisions)
    

    
    
    
    
    #print(X_train)
    print("data fetched")
    #sys.exit()
    Utils.writeToFile(logpath, "data fetched")

    realizations, realizationIndexes = source_USCO.readRealizations(featurePath, featureNum, maxNum = maxFeatureNum)

    
    
    
    train_start_time = datetime.now()
    #model = Model()

    qrts_P = QRTS_P()
    qrts_P.initialize(realizations, source_USCO)
        #**************************OneSlackSSVM_normal
    trainMethod = "one_slack_normal"
    w=qrts_P.train(trainMethod, verbose, C, tol, thread, max_iter, logpath,TrainQueries, TrainDecisions, beta)

    
    train_end_time = datetime.now()
    train_time = train_end_time  - train_start_time
    print(train_time)

    
    if pre_train is True:
        now = datetime.now()
        preTrainPath=path+"/pre_train/"+now.strftime("%d-%m-%Y-%H-%M-%S")+"/"
        if not os.path.exists(preTrainPath):
            os.makedirs(preTrainPath)
        Utils.save_pretrain(preTrainPath, w, realizationIndexes, featurePath)
        Utils.writeToFile(logpath, preTrainPath, toconsole = True)   
        preTrainPathResult = preTrainPath+"/result"
    
    Utils.writeToFile(logpath, "===============================================================", toconsole = True, preTrainPathResult = preTrainPathResult)
    
    
    Utils.writeToFile(logpath, "Testing ...", toconsole = True,preTrainPathResult = preTrainPathResult)
    
    #print(TestDecisions)

    Utils.writeToFile(logpath, sourceT+" "+targetT, toconsole = True, preTrainPathResult = preTrainPathResult) 
    Utils.writeToFile(logpath, "{}_{}".format(graph,graphType), toconsole = True, preTrainPathResult = preTrainPathResult)    
    Utils.writeToFile(logpath, "QRTS_P, featureNum: {}, trainMethod: {}, beta: {}".format(featureNum, trainMethod, beta), toconsole = True,preTrainPathResult = preTrainPathResult)
    Utils.writeToFile(logpath, "trainNum: {} ".format(trainNum, ), toconsole = True,preTrainPathResult = preTrainPathResult)
    Utils.writeToFile(logpath, "testNum:{} ".format(testNum), toconsole = True,preTrainPathResult = preTrainPathResult)
    Utils.writeToFile(logpath, "maxIter: {}, c: {} ".format(max_iter, C), toconsole = True,preTrainPathResult = preTrainPathResult)
    
    Utils.writeToFile(logpath, "=================================", toconsole = True, preTrainPathResult = preTrainPathResult)
    v = []
    for _ in range(6):
        v.append([])
    for TestSamples, TestQueries, TestDecisions in zip(TestSamples_s, TestQueries_s, TestDecisions_s):
        #Utils.writeToFile(logpath, "Making SI predictions ...", toconsole=True, preTrainPathResult=preTrainPathResult)
        predDecisions = target_USCO.solve_R_batch(TestQueries, w, realizations, n_jobs=thread, offset=None, trace = False)
        r = target_USCO.test(TestSamples, TestQueries, TestDecisions, predDecisions, thread, logpath = logpath, preTrainPathResult = preTrainPathResult )
        for i in range(6):
            v[i].append(r[i])
    v_mean = []
    for i in range(6):
        # print(v[i])
        v_mean.append(np.mean(np.array(v[i])))
    Utils.writeToFile(logpath, "-----------------------", toconsole=True,
                      preTrainPathResult=preTrainPathResult)
    output = "{} ({}), {} ({}), {}, {}".format(Utils.formatFloat(v_mean[0]),
                                               Utils.formatFloat(v_mean[1]),
                                               Utils.formatFloat(v_mean[2]),
                                               Utils.formatFloat(v_mean[3]), Utils.formatFloat(v_mean[4]),
                                               Utils.formatFloat(v_mean[5]))
    Utils.writeToFile(logpath, output, toconsole=True, preTrainPathResult=preTrainPathResult)

    Utils.writeToFile(logpath, "=================================", toconsole=True,
                      preTrainPathResult=preTrainPathResult)
    Utils.writeToFile(logpath, "All ones", toconsole=True, preTrainPathResult=preTrainPathResult)
    v = []
    for _ in range(6):
        v.append([])
    for TestSamples, TestQueries, TestDecisions in zip(TestSamples_s, TestQueries_s, TestDecisions_s):
        randomW = np.ones(featureNum)
        randPredDecisions = target_USCO.solve_R_batch(TestQueries, randomW, realizations, n_jobs=thread, offset=None, trace=False)
        r = target_USCO.test(TestSamples, TestQueries, TestDecisions, randPredDecisions, thread, logpath=logpath, preTrainPathResult=preTrainPathResult)
        for i in range(6):
            v[i].append(r[i])

    v_mean = []
    for i in range(6):
        v_mean.append(np.mean(np.array(v[i])))
    Utils.writeToFile(logpath, "-----------------------", toconsole=True,
                      preTrainPathResult=preTrainPathResult)
    output = "{} ({}), {} ({}), {}, {}".format(Utils.formatFloat(v_mean[0]),
                                               Utils.formatFloat(v_mean[1]),
                                               Utils.formatFloat(v_mean[2]),
                                               Utils.formatFloat(v_mean[3]), Utils.formatFloat(v_mean[4]),
                                               Utils.formatFloat(v_mean[5]))
    Utils.writeToFile(logpath, output, toconsole=True, preTrainPathResult=preTrainPathResult)
        #Utils.writeToFile(logpath, "-----------------------", toconsole=True, preTrainPathResult=preTrainPathResult)
    Utils.writeToFile(logpath, "=================================", toconsole=True, preTrainPathResult=preTrainPathResult)


    return "{}".format(now.strftime("%d-%m-%Y-%H-%M-%S"))

    
    
    
   
if __name__ == "__main__":
    main()
    