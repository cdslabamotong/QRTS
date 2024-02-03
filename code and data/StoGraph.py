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
from Utils import Dijkstra, Utils


class StoGraph(object):  
    class Node(object):
        def __init__(self,index):
            self.index = index
            self.neighbor = {}
            self.in_degree = 0
            self.out_degree = 0
        def myprint(self):
            print(self.index)
            for node in self.neighbor:
                print("{} {} {} {}".format(str(self.index), str(node) , str(self.neighbor[node][0]), str(self.neighbor[node][1])))       
    
    def __init__(self, path, vNum, graphType):
        self.nodes={}
        self.vNum = vNum
        self.EGraph = None
        self.graphType = graphType
        
        for v in range(self.vNum):
             node = self.Node(str(v))
             node.neighbor={}
             self.nodes[str(v)]=node
             
        file1 = open(path, 'r') 
        while True: 
            line = file1.readline() 
            if not line: 
                break          
            ints = line.split()
            if graphType == "Weibull":
                if len(ints) == 6:
                    node1 = ints[0]
                    node2 = ints[1]
                    alpha = int(ints[2])
                    beta = int(ints[3])
                    mean = float(ints[4]) # mean of Weibull
                    prob = float(ints[5]) # prob of edge
                else:
                    if len(ints) == 5:
                        node1 = ints[0]
                        node2 = ints[1]
                        alpha = int(ints[2])
                        beta = int(ints[3])
                        mean = float(ints[4])  # mean of Weibull
                        prob = 1  # prob of edge
                    else:
                        sys.exit("stoGraph wrong input format")
                
                if node1 in self.nodes:
                    if node2 not in self.nodes[node1].neighbor:
                        self.nodes[node1].neighbor[node2]=[alpha,beta, mean, prob]
                        self.nodes[node1].out_degree += 1
                        self.nodes[node2].in_degree += 1
                else:
                    sys.exit("non existing node") 
                    
                if node2 not in self.nodes:
                    sys.exit("non existing node") 
            
            else:
                if graphType == "Gaussian_1_10_100":
                    if len(ints) == 6:
                        node1 = ints[0]
                        node2 = ints[1]
                        mean = int(ints[2])
                        variance = int(ints[3])
                        mean = float(ints[4]) # mean of Weibull
                        prob = float(ints[5]) # prob of edge
                    else:
                        sys.exit("stoGraph wrong input format")

                    if node1 in self.nodes:
                       if node2 not in self.nodes[node1].neighbor:
                           self.nodes[node1].neighbor[node2]=[mean,variance, mean, prob]
                           self.nodes[node1].out_degree += 1
                           self.nodes[node2].in_degree += 1
                    else:
                        sys.exit("non existing node")

                    if node2 not in self.nodes:
                        sys.exit("non existing node")
                else:
                    sys.exit("no weight type")
        
        #create mean graph
        temp_nodes  =  copy.deepcopy(self.nodes) 
        for node in temp_nodes:
            for tonode in temp_nodes[node].neighbor:
                temp_nodes[node].neighbor[tonode]=self.nodes[node].neighbor[tonode][2];
        
        self.EGraph = Graph(self.vNum, temp_nodes)
        
        #create unit graph
        temp_nodes  =  copy.deepcopy(self.nodes) 
        for node in temp_nodes:
            for tonode in temp_nodes[node].neighbor:
                temp_nodes[node].neighbor[tonode]=1;
        
        self.unitGraph = Graph(self.vNum, temp_nodes)
        
        #create random graph
        temp_nodes  =  copy.deepcopy(self.nodes) 
        for node in temp_nodes:
            for tonode in temp_nodes[node].neighbor:
                temp_nodes[node].neighbor[tonode]=random.random()
        self.randomGraph = Graph(self.vNum, temp_nodes)

    def EgraphShortest(self, source, destination = None):
        return Dijkstra.dijkstra(self.EGraph,source)


    def genEdgeRealizations(self, outfolder, weightType):
        #raise NotImplementedError()
        cout = 0;
        for node in self.nodes:
            for tonode in self.nodes[node].neighbor:
                outpath = "{}{}".format(outfolder, cout)
                with open(outpath, 'w') as outfile:
                    if weightType == "true":
                        weight = self.sampleTrueWeight(node, tonode)
                    else:
                        if weightType == "unit":
                            weight = 1
                        else:
                            sys.exit("wrong weightType")
                    outfile.write(node + " ")
                    outfile.write(tonode + " ")
                    outfile.write(str(weight) + "\n")
                outfile.close()
                cout += 1

    def genMultiRealization(self, num, outfolder, edgeType, edgeParameter, weightType, weightParameter, startIndex, distance ):
        #raise NotImplementedError()       
        for cout in range(num):
            #print(cout)
            outpath = "{}{}".format(outfolder, cout+startIndex)
            self.genOneRealization(outpath, edgeType, edgeParameter, weightType, weightParameter, distance)


    def genMultiRealization_P(self, num, outfolder, edgeType, edgeParameter , weightType , weightParameter , thread = 1, startIndex = 0, distance = True ):
        #raise NotImplementedError()
        block_size =int (num/thread);
        p = multiprocessing.Pool(thread)
            #print("222")
        p.starmap(self.genMultiRealization, ((block_size, outfolder, edgeType, edgeParameter, weightType, weightParameter,startIndex+i*block_size, distance) for i in range(thread)) )
        #print("333")
        p.close()
        p.join()
        
        
            
    def genOneRealization(self, outpath, edgeType, edgeParameter, weightType, weightParameter, distance):
        '''
        Parameters
        ----------
        edgeType: true, threshold
        edgeParameter: none, keep_rate
        weightType: true, uniform, exp
        weightParameter: none, scale, scale
        distance

        Returns
        -------

        '''
        #raise NotImplementedError()
        graph = Graph(self.vNum, copy.deepcopy(self.nodes))
        with open(outpath, 'w') as outfile:
            for node in self.nodes:
                graph.nodes[node].neighbor={}
                for tonode in self.nodes[node].neighbor:
                    [ifEdge, weight] = self.getOneEdge(node, tonode, edgeType, edgeParameter, weightType, weightParameter)
                    if ifEdge == None or weight == None:
                        sys.exit("wrong edgeType or weightType")
                    if ifEdge:
                        graph.nodes[node].neighbor[tonode] = weight
                        outfile.write(node + " ")
                        outfile.write(tonode + " ")
                        outfile.write(str(weight) + "\n")
        outfile.close()
        
        if distance is True:
            with open(outpath+"_distance", 'w') as outfile:
                for node in self.nodes:                   
                    result = Dijkstra.dijkstra(graph, node);
                    for tonode in self.nodes:
                        if tonode in result:
                            string = result[tonode][0]+" "+result[tonode][1]+" "+result[tonode][2]+"\n" 
                            outfile.write(string) 
            outfile.close()

    def getOneEdge(self, node, tonode, edgeType, edgeParameter, weightType, weightParameter):
        ifEdge = None
        if edgeType=="true":
            ifEdge = self.sampleEdge(node,tonode)
        elif edgeType=="threshold":
            if (random.random() >= edgeParameter):
                ifEdge = False
            else:
                ifEdge = True

        weight = None
        if weightType == "true":
            weight = self.sampleTrueWeight(node, tonode)
        elif weightType == "uniform":
            weight = self.sampleUniform(weightParameter)
        elif weightType == "exp":
            weight = self.sampleExp(weightParameter)
        elif weightType == "WeibullRandom":
            alpha = random.random()+0.1
            lamb = random.random() * 10
            weight = Utils.getWeibull_alpha_lambda(alpha,lamb)

        return ifEdge, weight



    def sampleTrueWeight(self, fromNode, toNode):
        alpha = self.nodes[fromNode].neighbor[toNode][0]
        beta = self.nodes[fromNode].neighbor[toNode][1]
        return Utils.getWeibull(alpha, beta)
        #raise NotImplementedError()
        
    def sampleUniform(self, scale):
        return scale*random.random()
        #raise NotImplementedError()

    def sampleExp(self, scale):
        return scale*np.random.exponential(1,1)[0]
        #raise NotImplementedError()
    
    def sampleEdge(self, fromNode, toNode):
        if random.random() >= self.nodes[fromNode].neighbor[toNode][3]:
            return False
        else:
            return True
        
    def print(self):
        for node in self.nodes:
            self.nodes[node].myprint()
            
    #@staticmethod
    def GenStoGraph(self, outpath):
         with open(outpath, 'w') as outfile:
            for node in self.nodes:
                for toNode in self.nodes[node].neighbor:
                    alpha = str(self.nodes[node].neighbor[toNode][0])
                    beta = str(self.nodes[node].neighbor[toNode][1])
                    mean = str(self.nodes[node].neighbor[toNode][2])
                    #ber = str(1.0)
                    ber = str(1.0/self.nodes[toNode].in_degree)
                    string = node + " " + toNode +" "+alpha+" "+beta+" "+mean+" "+ber+"\n"
                    outfile.write(string)
         outfile.close()   
         
    def GenEmModel_Uniform(self, outpath):
        with open(outpath, 'w') as outfile:
           for node in self.nodes:
               for toNode in self.nodes[node].neighbor:
                   alpha = int(random.random()*10)+1
                   beta = int(random.random()*10)+1
                   mean = Utils.getWeibull(alpha, beta)
                   ber = random.random()/10
                   #ber = str(1.0/self.nodes[toNode].in_degree)
                   string = node + " " + toNode +" "+str(alpha)+" "+str(beta)+" "+str(mean)+" "+str(ber)+"\n"
                   outfile.write(string)
        outfile.close()   
        
    def GenEmModel_Approx(self, outpath, weight_q, prob_q):
        with open(outpath, 'w') as outfile:
           for node in self.nodes:
               for toNode in self.nodes[node].neighbor:
                   alpha = int(2*weight_q*self.nodes[node].neighbor[toNode][0]*random.random()+(1-weight_q)*self.nodes[node].neighbor[toNode][0])
                   beta = int(2*weight_q*self.nodes[node].neighbor[toNode][1]*random.random()+(1-weight_q)*self.nodes[node].neighbor[toNode][1])
                   mean = Utils.getWeibull(alpha, beta)
                   ber = 2*prob_q*self.nodes[node].neighbor[toNode][3]*random.random()+(1-prob_q)*self.nodes[node].neighbor[toNode][3]
                   #ber = str(1.0/self.nodes[toNode].in_degree)
                   string = node + " " + toNode +" "+str(alpha)+" "+str(beta)+" "+str(mean)+" "+str(ber)+"\n"
                   outfile.write(string)
        outfile.close()  
        
    def GenMixGaussian_1_10_100(self, outpath):
        means = [1,10,100]
        variances = [5]
        with open(outpath, 'w') as outfile:
           for node in self.nodes:
               for toNode in self.nodes[node].neighbor:
                   mean = random.choice(means)
                   variance = random.choice(variances)
                   #beta = int(2*weight_q*self.nodes[node].neighbor[toNode][1]*random.random()+(1-weight_q)*self.nodes[node].neighbor[toNode][1])
                   #mean = Utils.getWeibull(alpha, beta)
                   #ber = 2*prob_q*self.nodes[node].neighbor[toNode][3]*random.random()+(1-prob_q)*self.nodes[node].neighbor[toNode][3]
                   #ber = str(1.0/self.nodes[toNode].in_degree)
                   string = node + " " + toNode +" "+str(mean)+" "+str(variance)+" "+str(mean)+" "+"1"+"\n"
                   outfile.write(string)
        outfile.close()  
        
         #raise NotImplementedError()
    
                
class Graph(): 
 
    def __init__(self, vNum, nodes, type = None, path = None):
        self.vNum = vNum
        self.nodes = nodes
    
    def isConnected(self, node1, node2):
        checkedList=[]
        c_nodes=[]
        
        c_nodes.append(node1)
        #checkedList.append(node2)
        
        while len(c_nodes)>0:
            temp_node = []
            for node in c_nodes:
                for tonode in self.nodes[node].neighbor: 
                    if tonode == node2:
                        return True
                    if tonode not in checkedList:
                        temp_node.append(tonode) 
                checkedList.append(node) 
            c_nodes=copy.copy(temp_node)
        
        return False
    
    def print_(self):
        for node in self.nodes:
            for tonode in self.nodes[node].neighbor:
                print(node+" "+tonode+" "+str(self.nodes[node].neighbor[tonode]))
                
    def pathLength(self, x, y):
        #print(x)
        #print(y)
        if y is None:
            return None
        length = 0
        if y[0]!= x[0] or y[-1]!=x[1]:
            print(x)
            print(y)
            sys.exit("path y not for x") 
            return None;
        else:
            for i in range(len(y)-1):
                if y[i] in self.nodes and y[i+1] in self.nodes[y[i]].neighbor:
                    length += float(self.nodes[y[i]].neighbor[y[i+1]])
                else:
                    print(y[i]+" "+y[i+1])
                    print(y)
                    sys.exit("edge not existing") 
                    return None;
            return length
        #self.adjmatrix = {};


if __name__ == "__main__":
    #pass
    #graph,  graphType, vNum = "kro_2", 'Weibull', 1024
    graph,  graphType, vNum = "ba", 'Weibull', 501

    #stoGraph=StoGraph("data/kro/kro_model", vnum, "Weibull")
    #stoGraph.GenMixGaussian_1_10_100("data/kro/kro_model_Gaussian_1_10_100")
    
    #stoGraph=StoGraph("data/{}/{}_stoGraph_Gaussian_1_10_100".format(dataname,dataname), vnum, "Gaussian")
    #stoGraph.GenEmModel_Uniform("data/{}/{}_model_em_uniform".format(dataname,dataname))

    #stoGraph.GenMixGaussian_1_10_100("data/{}/{}_stoGraph_Gaussian_1_10_100".format(dataname,dataname))
    
    
    stoGraph=StoGraph("data/{}/{}_{}_stoGraph".format(graph,graph,graphType), vNum, "Weibull")
    realizationType = "WeibullRandom"
    #stoGraph.genMultiRealization_P(10000, "data/{}/features/{}_10000/".format(graph, realizationType), "true", None, realizationType, 1, thread=5, startIndex=0, distance=False)
    stoGraph.genEdgeRealizations("data/{}/features/edge_unit/".format(graph),'unit')
    pass