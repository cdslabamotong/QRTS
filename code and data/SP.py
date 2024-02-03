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



    
    
class SP_USCO(USCO):
    
    class Realization(object):
        def __init__(self):
            pass

        def initialize_file(self, graphPath, vNum):
            self.weightMatrix = {}
            #self.distance = {}
            self.nodes = set()
            self.vNum = vNum

            for v in range(self.vNum):
                # node = self.Node(str(v))
                # node.neighbor={}
                self.weightMatrix[str(v)] = {}
                self.nodes.add(str(v))

            file1 = open(graphPath, 'r')
            while True:
                line = file1.readline()
                if not line:
                    break
                strings = line.split()
                node1 = (strings[0])
                node2 = (strings[1])
                weight = float(strings[2])

                if node1 in self.weightMatrix:
                    self.weightMatrix[node1][node2] = weight

                else:
                    sys.exit("non existing node")

                if node2 not in self.nodes:
                    sys.exit("non existing node")

        def initialize_matrix(self, matrix, vNum):
            self.weightMatrix = {}
            # self.distance = {}
            self.nodes = set()
            self.vNum = vNum

            for v in range(self.vNum):
                # node = self.Node(str(v))
                # node.neighbor={}
                self.weightMatrix[str(v)] = {}
                self.nodes.add(str(v))

            for node1 in matrix:
                for node2 in matrix[node1].neighbor:
                    if node1 in self.weightMatrix:
                        self.weightMatrix[node1][node2] =  matrix[node1].neighbor[node2]
                    else:
                        sys.exit("non existing node")
                    if node2 not in self.nodes:
                        sys.exit("non existing node")
        # def __init__(self, graphPath, vNum):
        #     self.weightMatrix={}
        #     #self.distance = {}
        #     self.nodes=set()
        #     self.vNum = vNum
        #
        #     for v in range(self.vNum):
        #          #node = self.Node(str(v))
        #          #node.neighbor={}
        #          self.weightMatrix[str(v)]={}
        #          #self.distance[str(v)]={}
        #          self.nodes.add(str(v))
        #
        #     file1 = open(graphPath, 'r')
        #     while True:
        #         line = file1.readline()
        #         if not line:
        #             break
        #         strings = line.split()
        #         node1 = (strings[0])
        #         node2 = (strings[1])
        #         weight = float(strings[2])
        #
        #         if node1 in self.weightMatrix:
        #             self.weightMatrix[node1][node2]=weight
        #         else:
        #             sys.exit("non existing node")
        #
        #         if node2 not in self.nodes:
        #             sys.exit("non existing node")
        #
        #     return
        #     raise NotImplementedError()
            
    class Query(object):
            def __init__(self, x_pair):
                self.x_pair=x_pair
                #print(x_set)
                #self.budget=budget
            
    class Sample(object):
            
        def __init__(self, query, y_path, length):
            self.query=query
            self.decision=y_path
            self.length=length
            #raise NotImplementedError()
            
        def print(self):
            #x_set = self.query.x_set
            print(self.query.x_pair)
            print(self.y_path)
            print(self.length)
            pass
            
            raise NotImplementedError()
    
    def __init__(self, stoGraph, trueAllPairs_path = None):
        self.stoGraph = stoGraph 
        
        
        #self.trueAllPairs = 
        #self.realizations, self.realizationIndexes = self.readRealizations(realizationPath, realizationNum)
        #self.realizations = None
        #self.realizationIndexes = None
        #self.pairs = None
        #self.samples = self.readSamples(samplePath, sampleNum)
        #self.samples = None
        #self.vNum=vNum
        
    #def readRealizations(self, realizationPath, realizationNum)
    
    def kernel(self, realization, query, y):
        if len(y) < 2 :
            sys.exit("empty path")
                    
        if query.x_pair[0]==y[0] and query.x_pair[1]==y[-1]:
            path_length = 0;
            for i in range(len(y)-1):
                node1=y[i]
                node2=y[i+1]
                if node2 in realization.weightMatrix[node1]:
                    path_length=path_length+realization.weightMatrix[node1][node2]
                else:
                    pass
                    #sys.exit("edge not in realization")
            return -path_length
        else:
            print(query.x_pair)
            print(y)
            sys.exit("not a path for x")
            return math.inf    
        raise NotImplementedError()
        
    
    
   
        
    def solve_R(self, realizations, W, x, E_graph=None):

        new_nodes=copy.deepcopy(self.stoGraph.nodes)
        
        for node1 in new_nodes:
            for node2 in new_nodes[node1].neighbor:
                new_nodes[node1].neighbor[node2]=0
        for (realization, w) in zip(realizations, W):
             for node1 in realization.weightMatrix:
                 for node2 in realization.weightMatrix[node1]:
                     new_nodes[node1].neighbor[node2] += realization.weightMatrix[node1][node2]
                     
        new_graph = Graph(self.stoGraph.vNum,new_nodes)
        
        result = Dijkstra.dijkstra(new_graph, x.x_pair[0])
        #print(x)
        #print(result[x[1]])
        return result[x.x_pair[1]][3:]
        
        raise NotImplementedError()
    
    def solve_R_batch(self, X, W, realizations, n_jobs, offset = None, trace = True):
        #print("inference")
        if trace:
            print("solve_R_batch RUNNING",format(n_jobs))
        new_nodes=copy.deepcopy(self.stoGraph.nodes)
        
        for node1 in new_nodes:
            for node2 in new_nodes[node1].neighbor:
                new_nodes[node1].neighbor[node2]=0
                
        for (realization, w) in zip(realizations, W):
             for node1 in realization.weightMatrix:
                 for node2 in realization.weightMatrix[node1]:
                     new_nodes[node1].neighbor[node2] += w*realization.weightMatrix[node1][node2]
                     
        new_graph = Graph(self.stoGraph.vNum,new_nodes)
        #new_graph.print_()
        
        #sys.exit("stop")
        
        nodeSet = []
        for x in X:
            if x.x_pair[0] not in nodeSet:
                nodeSet.append(x.x_pair[0])
            
        #X_pathRestuls = {}
        
        #if n_jobs == 1:
         #   result =[]
          #  for x in X:
           #     result.append(self.solve_R(realizations, W, x))
            #print("solve_R_batch DONE")
            #return result
        #else:
            #print(n_jobs)
            #results={}
            #p = multiprocessing.Pool(n_jobs)
            #print("222")
            #results=p.starmap(self.solve_R, ((realizations, W, x) for x in X))
            #p.close()
            #p.join()

        results=[]
        p = multiprocessing.Pool(n_jobs)
        results=p.starmap(Dijkstra.dijkstra, ((new_graph, x) for x in nodeSet))
        p.close()
        p.join()
        
        '''
        results =[]
        p = multiprocessing.Pool(n_jobs)
        block_size =int(len(nodeSet)/n_jobs)
        Ys=p.starmap(Dijkstra.dijkstra_batch, ((new_graph, nodeSet[i*block_size:min([len(nodeSet),(i+1)*block_size])]) for i in range(n_jobs) ))
        p.close()
        p.join()
        for block in Ys:
            results.extend(block)
        '''
        
        X_pathRestuls = {}
        #print("{} {}".format(len(nodeSet),len(results)))
        for node, result in zip(nodeSet, results):
            X_pathRestuls[node]=result
        #print("solve_R_batch DONE")
        pathResult = []
        for x in X:
            pathResult.append(X_pathRestuls[x.x_pair[0]][x.x_pair[1]][3:])
        if trace:
            print("solve_R_batch DONe")
        return pathResult 
   
    def solveTrue(self, x):
        '''
        Define how to compute the true solution
        '''
        raise NotImplementedError()

    def solve_R_batch_random(self, X, P_realizations, n_jobs=1, offset=None, trace=True):
        if n_jobs == 1:
            Y = []
            for x in X:
                Y.append(self.solve_R_one_random(x, P_realizations))
        else:
            p = multiprocessing.Pool(n_jobs)
            Y = p.starmap(self.solve_R_one_random, [(x, P_realizations) for x in X], chunksize =100)
            p.close()
            # p.terminate()
            p.join()

        return Y

    def solve_R_one_random(self, x, P_realizations):

        new_nodes = copy.deepcopy(self.stoGraph.nodes)

        for node1 in new_nodes:
            for node2 in new_nodes[node1].neighbor:
                alpha = random.random() + 0.1
                lamb = random.random() * 10
                mean = Utils.getWeibull_alpha_lambda(alpha, lamb)
                new_nodes[node1].neighbor[node2] = max(mean,0)


        new_graph = Graph(self.stoGraph.vNum, new_nodes)

        result = Dijkstra.dijkstra(new_graph, x.x_pair[0])
        # print(x)
        # print(result[x[1]])
        return result[x.x_pair[1]][3:]
    
    def genQuery(self):
        raise NotImplementedError()
        
            
       
    def genSamples_P(self, Wpath = None):
        strings = []        
        for node in self.stoGraph.nodes:
                result = Dijkstra.dijkstra(self.stoGraph.EGraph, node);
                for tonode in self.stoGraph.nodes:
                    if tonode in result:
                        string = ""
                        for i in result[tonode]:
                            #print(" "+i)
                            string += i
                            string += " "
                        string += "\n"
                        if (result[tonode][0]!=result[tonode][1] and len(result)>5):
                            strings.append(string)
                            #print(string)
                            
                print(node)
        random.shuffle(strings)
        if Wpath != None:
            with open(Wpath, 'w') as outfile:
                for string in strings:
                    outfile.write(string)
            outfile.close()
        #raise NotImplementedError()

    def genSamples_P(self, n_jobs = 1, Wpath=None):
        strings = []
        if n_jobs ==1 :
            for node in self.stoGraph.nodes:
                result = Dijkstra.dijkstra(self.stoGraph.EGraph, node);
                for tonode in self.stoGraph.nodes:
                    if tonode in result:
                        string = ""
                        for i in result[tonode]:
                            # print(" "+i)
                            string += i
                            string += " "
                        string += "\n"
                        if (result[tonode][0] != result[tonode][1] and len(result) > 5):
                            strings.append(string)
                            # print(string)

                print(node)
        else:
            p = multiprocessing.Pool(n_jobs)
            results = p.starmap(Dijkstra.dijkstra, [(self.stoGraph.EGraph, node) for node in self.stoGraph.nodes])
            p.close()
            p.join()
            for node, result in zip(self.stoGraph.nodes, results):
                for tonode in self.stoGraph.nodes:
                    if tonode in result:
                        string = ""
                        for i in result[tonode]:
                            # print(" "+i)
                            string += i
                            string += " "
                        string += "\n"
                        if (result[tonode][0] != result[tonode][1] and len(result) > 5):
                            strings.append(string)




        random.shuffle(strings)
        if Wpath != None:
            with open(Wpath, 'w') as outfile:
                for string in strings:
                    outfile.write(string)
            outfile.close()
    
    def readSamples(self, Rpath, num, Max, isRandom = True, RmaxSize = False):
        lineNums=(np.random.permutation(Max))[0:num] 
        #lineNums.sort()
        file1 = open(Rpath, 'r') 
        lineNum = 0
        queries = []
        decisions = []
        samples = []
        lengths = []
        #maxSize = 0
        
        while True:
            line = file1.readline() 
            if not line: 
                break 
            strings=line.split()
            if lineNum in lineNums:
                #print(str(lineNum))
                #print(trainLineNums)
                x_pair = [strings[0],strings[1]]
               
                decision = strings[3:]    
                length = float(strings[2])
                query=self.Query(x_pair)
                #print("query generated")
                sample = self.Sample(query, decision, length)
                #print("sample generated")
                
                queries.append(query)
                decisions.append(decision)
                samples.append(sample) 
                lengths.append(length)
                
                #if len(x_set) > maxSize:
                #    maxSize = len(x_set)
                #if len(decision) > maxSize:
                #    maxSize = len(decision)   
                
            lineNum += 1 
        
        #if RmaxSize  is True:
        #    return samples, queries, decisions, maxSize
        #else:
        return samples, queries, decisions
        
    
    def readRealizations(self, Rfolder, realizationsNum, indexes = None, realizationRandom = True, maxNum = None ):
        #print(realizationsNum)
        realizations = []
        realizationIndexes = []
        if indexes is not None:
             realizationIndexes=indexes
        else:  
            if realizationRandom:
                if maxNum is None:
                    sys.exit("maxNum for specified when realizationRandom = True")
                    
                lineNums=(np.random.permutation(maxNum))[0:realizationsNum]
                realizationIndexes=lineNums
                #print("lineNums: {}".format(lineNums))
            else:
                for i in range(realizationsNum):
                    realizationIndexes.append(i)
        
        for i in realizationIndexes:             
            path_graph="{}/{}".format(Rfolder, i)
            #path_dis="{}/{}_distance".format(Rfolder, i)
            realization=self.Realization()
            realization.initialize_file(path_graph, self.stoGraph.vNum)
            realizations.append(realization)
        #print(realizationIndexes)
        print("readRealizations done")
        return realizations, realizationIndexes
    
    def test(self, TestSamples, TestQueries, TestDecisions, predDecisions, n_jobs, logpath = None, preTrainPathResult = None):
        trueLengths = []
        predLengths = []
        ratios = []
        re_ratios = []
        #inter = []
        
        #p = multiprocessing.Pool(n_jobs)
        #block_size =int (len(TestQueries)/n_jobs)
        #Ys=p.starmap(self.inf_block, ((TestQueries[i*block_size:min([len(TestQueries),(i+1)*block_size])], predDecisions[i*block_size:min([len(TestQueries),(i+1)*block_size])], 1000) for i in range(n_jobs) ))
        #p.close()
        #p.join()
        #decisions = []
        #for inf_block in Ys:
        #    predInfs.extend(inf_block)
                
 
        for (sample, predDecision) in zip(TestSamples, predDecisions):            
            predLength = self.stoGraph.EGraph.pathLength(sample.query.x_pair, predDecision)
            trueLenth = self.stoGraph.EGraph.pathLength(sample.query.x_pair, sample.decision)
            if predLength/trueLenth < 1:
                print("{} {} {}".format(predLength, trueLenth, Utils.formatFloat(predLength/trueLenth)))

            trueLengths.append(sample.length)
            predLengths.append(predLength)
            ratios.append(predLength/sample.length)
            re_ratios.append(sample.length/predLength)
        
        
        mean_ratios=np.mean(np.array(ratios))
        std_ratios=np.std(np.array(ratios))

        mean_re_ratios = np.mean(np.array(re_ratios))
        std_re_ratios = np.std(np.array(re_ratios))

        
        mean_trueLengths=np.mean(np.array(trueLengths))
        std_trueLengths=np.std(np.array(trueLengths))
        
        mean_predLengths=np.mean(np.array(predLengths))
        std_predLengths=np.std(np.array(predLengths))
        


        

        #output = "Pred Total Length/True Total Length: {} ({}) / {} ({})".format(Utils.formatFloat(mean_predLengths), Utils.formatFloat(std_predLengths), Utils.formatFloat(mean_trueLengths), Utils.formatFloat(std_trueLengths))
        #Utils.writeToFile(logpath, output, toconsole = True,preTrainPathResult = preTrainPathResult)


        #output = "Approx ratio: {} ({})   Approx re_ratio: {} ({})".format(Utils.formatFloat(mean_ratios), Utils.formatFloat(std_ratios),Utils.formatFloat(mean_re_ratios), Utils.formatFloat(std_re_ratios))
        #Utils.writeToFile(logpath, output, toconsole=True, preTrainPathResult=preTrainPathResult)



        #output = "Length ratio: {}   Length re_ratio: {} ".format(Utils.formatFloat(mean_predLengths/mean_trueLengths), Utils.formatFloat(mean_trueLengths / mean_predLengths))
        #Utils.writeToFile(logpath, output, toconsole=True, preTrainPathResult=preTrainPathResult)

        output = "{} ({}), {} ({}), {}, {}".format(Utils.formatFloat(mean_ratios),
                                                   Utils.formatFloat(std_ratios),
                                                   Utils.formatFloat(mean_re_ratios),
                                                   Utils.formatFloat(std_re_ratios),
                                                   Utils.formatFloat(mean_predLengths/mean_trueLengths), Utils.formatFloat(mean_trueLengths / mean_predLengths))
        Utils.writeToFile(logpath, output, toconsole=True, preTrainPathResult=preTrainPathResult)

        return  mean_predLengths/mean_trueLengths, mean_trueLengths / mean_predLengths, mean_ratios, std_ratios, mean_re_ratios, std_re_ratios
   
   



def main():

    graph = "ba"
    graphType = 'Weibull'
    task = "SP"
    vnum=501
    stoGraph=StoGraph("data/{}/{}_{}_stoGraph".format(graph,graph, graphType), vnum, graphType)
    source_USCO = SP_USCO(stoGraph)
    source_USCO.genSamples_P(n_jobs = 10, Wpath="data/{}/{}_{}_{}_samples".format(graph,graph, graphType, task))
    #source_USCO.genSamples()

    '''
    dataname = "kro_Gaussian_1_10_100"
    vnum = 1024
    stoGraph = StoGraph("data/{}/{}_stoGraph".format(dataname, dataname), vnum, "Gaussian_1_10_100")
    source_USCO = SP_USCO(stoGraph)
    source_USCO.genSamples("data/{}/{}_samples".format(dataname, dataname))
    '''
    pass
#g = Graph(9)
if __name__ == "__main__":
    pass
    main()
    