# -*- coding: utf-8 -*-
"""
# License: BSD 3-clause
"""
import numpy as np
import sys
import random
# import time
import copy
import heapq
import multiprocessing
from multiprocessing import Process
from Utils import Utils
from USCO import USCO
from scipy.stats import powerlaw
from StoGraph import StoGraph
from itertools import chain

from networkx.utils import pairwise, not_implemented_for
import networkx as nx

#__all__ = ["metric_closure", "steiner_tree"]

class Steiner(object):

    def __init__(self):
        self.G = None
        self.M = None

    def initialize_G(self, G):
        self.G = G

    def initialize_M(self, M):
        self.M = M
    #@not_implemented_for("directed")
    def metric_closure(self, G, weight="weight"):
        """Return the metric closure of a graph.

        The metric closure of a graph *G* is the complete graph in which each edge
        is weighted by the shortest path distance between the nodes in *G* .

        Parameters
        ----------
        G : NetworkX graph

        Returns
        -------
        NetworkX graph
            Metric closure of the graph `G`.

        """
        M = nx.Graph()

        Gnodes = set(G)

        # check for connected graph while processing first node
        all_paths_iter = nx.all_pairs_dijkstra(G, weight=weight)
        u, (distance, path) = next(all_paths_iter)
        if Gnodes - set(distance):
            msg = "G is not a connected graph. metric_closure is not defined."
            raise nx.NetworkXError(msg)
        Gnodes.remove(u)
        for v in Gnodes:
            M.add_edge(u, v, distance=distance[v], path=path[v])

        # first node done -- now process the rest
        for u, (distance, path) in all_paths_iter:
            Gnodes.remove(u)
            for v in Gnodes:
                M.add_edge(u, v, distance=distance[v], path=path[v])

        return M

    #@not_implemented_for("directed")
    def steiner_tree_batch(self, terminal_nodes_s, n_jobs=1, weight="weight"):
        n_jobs = 1
        #print("steiner_tree_batch RUNNING", format(n_jobs))
        M = self.metric_closure(self.G, weight=weight)
        self.M = M
        results = []

        if n_jobs ==1:
            for terminal_nodes in terminal_nodes_s:
                T = self.steiner_tree_withMetricClosure(terminal_nodes)
                results.append(T)
        else:
            pass
            #self.steiner_tree_withMetricClosure(terminal_nodes_s[0])
            # print("111")
            # p = multiprocessing.Pool(n_jobs)
            # print(type(terminal_nodes_s))
            # results = p.starmap(self.steiner_tree_withMetricClosure, [(terminal_nodes_s[0],),(terminal_nodes_s[1],)])
            # p.close()
            # p.join()
        #     procs = []
        #     for terminal_nodes in terminal_nodes_s:
        #         # print(name)
        #         proc = Process(target=self.steiner_tree_withMetricClosure, args=(terminal_nodes,))
        #         procs.append(proc)
        #         proc.start()
        #
        #         # complete the processes
        #     for proc in procs:
        #         proc.join()
        # sys.exit()
        return results

    def steiner_tree_withMetricClosure(self, terminal_nodes):
        H = self.M.subgraph(terminal_nodes)
        # Use the 'distance' attribute of each edge provided by M.
        mst_edges = nx.minimum_spanning_edges(H, weight="distance", data=True)
        # Create an iterator over each edge in each shortest path; repeats are okay
        edges = chain.from_iterable(pairwise(d["path"]) for u, v, d in mst_edges)
        # For multigraph we should add the minimal weight edge keys
        if self.G.is_multigraph():
            edges = (
                (u, v, min(self.G[u][v], key=lambda k: self.G[u][v][k][weight])) for u, v in edges
            )
        T = self.G.edge_subgraph(edges)
        return T

    def steiner_tree(self, G, terminal_nodes, weight="weight"):
        """Return an approximation to the minimum Steiner tree of a graph.

        The minimum Steiner tree of `G` w.r.t a set of `terminal_nodes`
        is a tree within `G` that spans those nodes and has minimum size
        (sum of edge weights) among all such trees.

        The minimum Steiner tree can be approximated by computing the minimum
        spanning tree of the subgraph of the metric closure of *G* induced by the
        terminal nodes, where the metric closure of *G* is the complete graph in
        which each edge is weighted by the shortest path distance between the
        nodes in *G* .
        This algorithm produces a tree whose weight is within a (2 - (2 / t))
        factor of the weight of the optimal Steiner tree where *t* is number of
        terminal nodes.

        Parameters
        ----------
        G : NetworkX graph

        terminal_nodes : list
             A list of terminal nodes for which minimum steiner tree is
             to be found.

        Returns
        -------
        NetworkX graph
            Approximation to the minimum steiner tree of `G` induced by
            `terminal_nodes` .

        Notes
        -----
        For multigraphs, the edge between two nodes with minimum weight is the
        edge put into the Steiner tree.


        References
        ----------
        .. [1] Steiner_tree_problem on Wikipedia.
           https://en.wikipedia.org/wiki/Steiner_tree_problem
        """
        # H is the subgraph induced by terminal_nodes in the metric closure M of G.
        #print("here")
        M = self.metric_closure(G, weight=weight)
        #print("here again")
        H = M.subgraph(terminal_nodes)
        # Use the 'distance' attribute of each edge provided by M.
        mst_edges = nx.minimum_spanning_edges(H, weight="distance", data=True)
        # Create an iterator over each edge in each shortest path; repeats are okay
        edges = chain.from_iterable(pairwise(d["path"]) for u, v, d in mst_edges)
        # For multigraph we should add the minimal weight edge keys
        if G.is_multigraph():
            edges = (
                (u, v, min(G[u][v], key=lambda k: G[u][v][k][weight])) for u, v in edges
            )
        T = G.edge_subgraph(edges)
        return T

    def con_edgelist(self,res):
        pair = []
        for line in nx.generate_edgelist(res, data=False):
            x = line.split()
            e = [0, 0]
            e[0] = int(x[0])
            e[1] = int(x[1])
            pair.append(e)
        pair.sort(key=lambda row: (row[0], row[1]), reverse=False)
        return pair


class Steiner_USCO(USCO):
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

        def print(self):
            count = 0
            for node in self.nodes:
                for tonode in self.weightMatrix[node]:
                    count += 1
            print(count)

    class Sample(object):

        class Query(object):
              def __init__(self, a_set):
                self.a_set = a_set

        def __init__(self, query, p_set, length):
            self.query = query
            self.decision = p_set
            self.length = length
        def print(self):
            # x_set = self.query.x_set
            print(self.query.a_set)
            print(self.decision)
            pass

    # def initialize(self):
    #     self.useModel = True

    def kernel(self, realization, query, decision):
        x = query.a_set
        y = decision
        length = 0
        for j in range((len(y)//2)):
            if y[2*j] in realization.weightMatrix and y[2 * j + 1] in realization.weightMatrix[y[2 * j]]:
                length += realization.weightMatrix[y[2 * j]][y[2 * j + 1]]
            elif y[2*j + 1] in realization.weightMatrix and y[2 * j] in realization.weightMatrix[y[2 * j + 1]]:
                length += realization.weightMatrix[y[2 * j + 1]][y[2 * j]]
            else :
                #sys.exit("edge not existing {}, {}".format(y[2*j],y[2*j + 1]))
                #return -1
                pass
        return -length


        # raise NotImplementedError()

    #def computeFeature_batch(self, realizations, X, Y, n_jobs = 1):
        # joint_feature_ = np.zeros(len(realizations))
        # p = multiprocessing.Pool(n_jobs)
        # Ys = p.starmap(self.computeFeature, [(realizations, X[i], Y[i]) for i in range(len(X))])
        # p.close()
        # p.join()
        #
        # for feature in Ys:
        #     joint_feature_ += np.array(feature)
        #
        # return joint_feature_

        #raise NotImplementedError()

    def solve_R(self, realizations, W, query):
        ngraph = nx.Graph()
        nodes = copy.deepcopy(self.stoGraph.nodes)
        for node in nodes:
            ngraph.add_node(node)
        for node in nodes:
            for tonode in nodes[node].neighbor:
                temp = 0
                for i in range(len(W)):
                    if W[i] > 0:
                        temp += W[i] * realizations[i].weightMatrix[node][tonode]
                ngraph.add_edge(node, tonode, weight=temp)
        ngraph.remove_node('0')
        result = []
        temp_res = steiner_tree(ngraph, query.a_set)
        list_res = con_edgelist(temp_res)
        true_res = []
        for pair in list_res:
                true_res.append(str(pair[0]))
                true_res.append(str(pair[1]))
        result.append(true_res)
        while(len(result)==1):
            result = result[0]
        return result

    def solve_R_withGraph(self, ngraph, query):
        #print("solve_R_withGraph Running")
        result = []
        temp_res = steiner_tree(ngraph, query.a_set)
        list_res = con_edgelist(temp_res)
        true_res = []
        for pair in list_res:
            true_res.append(str(pair[0]))
            true_res.append(str(pair[1]))
        result.append(true_res)
        while (len(result) == 1):
            result = result[0]
        #print("solve_R_withGraph Done ")
        return result

    def solve_R_batch_random(self, X, P_realizations, n_jobs=1, offset=None, trace=True):
        if n_jobs == 1:
            Y = []
            for x in X:
                Y.append(self.solve_R_one_random(x, P_realizations))
        else:
            p = multiprocessing.Pool(n_jobs)
            Y = p.starmap(self.solve_R_one_random, [(x, P_realizations) for x in X])
            p.close()
            # p.terminate()
            p.join()

        return Y

    def solve_R_one_random(self, x, P_realizations):

        ngraph = nx.Graph()
        nodes = copy.deepcopy(self.stoGraph.nodes)
        for node in nodes:
            ngraph.add_node(node)


        for i in range(len(P_realizations)):
            for node in P_realizations[i].weightMatrix:
                for tonode in P_realizations[i].weightMatrix[node]:
                    alpha = random.random() + 0.1
                    lamb = random.random() * 10
                    mean = Utils.getWeibull_alpha_lambda(alpha, lamb)
                    ngraph.add_edge(node, tonode, weight=mean)


        ngraph.remove_node('0')

        #terminal_nodes_s = []
        #terminal_nodes_s.append(x.a_set)

        steiner = Steiner()
        steiner.initialize_G(ngraph)

        T = steiner.steiner_tree(ngraph, x.a_set)

        result = []
        list_res = steiner.con_edgelist(T)
        true_res = []
        for pair in list_res:
            true_res.append(str(pair[0]))
            true_res.append(str(pair[1]))
        result.append(true_res)
        while (len(result) == 1):
            result = result[0]


        return result

    def solve_R_batch(self, X, W, realizations, n_jobs=1, offset=None, trace = True):
        # print("inference")
        if trace:
            print("solve_R_batch Steiner RUNNING", format(n_jobs))

        ngraph = nx.Graph()
        nodes = copy.deepcopy(self.stoGraph.nodes)
        for node in nodes:
            ngraph.add_node(node)
        for node in nodes:
            for tonode in nodes[node].neighbor:
                temp = 0
                for i in range(len(W)):
                    if W[i] > 0:
                        if node in realizations[i].weightMatrix and tonode in realizations[i].weightMatrix[node]:
                            temp += W[i] * realizations[i].weightMatrix[node][tonode]
                ngraph.add_edge(node, tonode, weight=temp)
        ngraph.remove_node('0')

        terminal_nodes_s = []
        for x in X:
            terminal_nodes_s.append(x.a_set)

        steiner = Steiner()
        steiner.initialize_G(ngraph)

        Ts = steiner.steiner_tree_batch(terminal_nodes_s, n_jobs = n_jobs)
        results = []
        for T in Ts:
            result = []
            list_res = steiner.con_edgelist(T)
            true_res = []
            for pair in list_res:
                true_res.append(str(pair[0]))
                true_res.append(str(pair[1]))
            result.append(true_res)
            while (len(result) == 1):
                result = result[0]
            results.append(result)


        #results = []
        #p = multiprocessing.Pool(n_jobs)
        #results = p.starmap(self.solve_R_withGraph, ((ngraph, x) for x in X))
        #p.close()
        #p.join()

        #if trace:
            #print("solve_R_batch DONe")

        return results

    def solveTrue(self, query, graph):
        length = 0
        result = steiner_tree(graph, query)
        te = nx.get_edge_attributes(result, "weight").values()
        for element in te:
            length = length + element
        list_res = con_edgelist(result)
        return list_res, length

    def solveTrue_margin(self, query, decision, decision_new, times=1000):
        raise NotImplementedError()
        # return self.inf(query, decision.union(decision_new),times = times)-self.inf(query, decision, times = times)
        # pass


    def genQuery(self):
        raise NotImplementedError()

    def genNxGraph(self):
        g = nx.Graph()
        for i in range(1, 1024):
            g.add_node(str(i))
        infile = open("data/kro/kro_model", 'r')
        while True:
            line = infile.readline()
            if not line:
                infile.close()
                break
            ints = line.split()
            i = ints[0]
            j = ints[1]
            w = float(ints[4])
            g.add_edge(str(i), str(j), weight=w)
        return g

    def genSamples(self, Wpath, scale_x, scale_y, num, thread):
        samples = []
        X_size = powerlaw.rvs(2.5, scale=scale_x, size=num)
        graph = self.genNxGraph()
        if thread == 1:
            for x_size in X_size:
                sample = self.genSample(int(x_size) + 5, graph)
                samples.append(sample)
                # print(len(samples))
        else:
            p = multiprocessing.Pool(thread)
            samples = p.starmap(self.genSample, ((int(x_size) + 5, graph) for x_size in X_size))
            p.close()
            p.join()
        # count = 0

        with open(Wpath, 'w') as outfile:
            for sample in samples:
                string = ""
                for v in sample.query:
                    string = string + v + " "
                string = string + "|"
                # string = ""
                for v in sample.decision:
                    string = string + str(v[0]) + " " + str(v[1]) + " "
                string = string + "|"

                # string = ""
                string = string + str(sample.length) + "\n"
                outfile.write(string)

                # print(string)

        outfile.close()

    def genSample(self, x_size, graph):
        if x_size > self.stoGraph.vNum:
            sys.exit("x or y size too large")

        nodes = list(self.stoGraph.nodes.keys())
        random.shuffle(nodes)
        # print(new_topics)
        query = list(nodes[0:x_size])
        decision, length = self.solveTrue(query,graph)
        sample = self.Sample(query, decision, length)
        print("genSample Done {}".format(x_size))

        # sample.print()
        return sample
        # raise NotImplementedError()

    def readSamples(self, Rpath, num, Max, isRandom=True, RmaxSize=False):
        lineNums = (np.random.permutation(Max))[0:num]

        # lineNums.sort()
        file1 = open(Rpath, 'r')
        lineNum = 0
        queries = []
        decisions = []
        samples = []
        maxSize = 0

        while True:
            line = file1.readline()
            if not line:
                break
            strings = line.split('|')
            if lineNum in lineNums:
                # print(str(lineNum))
                # print(trainLineNums)
                a_set = list(strings[0].split())
                decision = list(strings[1].split())
                length = float(strings[2])
                query = self.Sample.Query(a_set)
                # print("query generated")
                sample = self.Sample(query,decision,length)
                # print("sample generated")

                queries.append(query)
                decisions.append(decision)
                samples.append(sample)


            lineNum += 1

        if RmaxSize is True:
            return samples, queries, decisions, maxSize
        else:
            return samples, queries, decisions
        # raise NotImplementedError()

    def readRealizations(self, Rfolder, realizationsNum, indexes=None, realizationRandom=True, maxNum=None):
        realizations = []
        realizationIndexes = []
        if indexes is not None:
            realizationIndexes = indexes
        else:
            if realizationRandom:
                if maxNum is None:
                    sys.exit("maxNum for specified when realizationRandom = True")

                lineNums = (np.random.permutation(maxNum))[0:realizationsNum]
                realizationIndexes = lineNums
                # print("lineNums: {}".format(lineNums))
            else:
                for i in range(realizationsNum):
                    realizationIndexes.append(i)

        for i in realizationIndexes:
            path_graph = "{}/{}".format(Rfolder, i)
            realization = self.Realization()
            realization.initialize_file(path_graph, self.stoGraph.vNum)
            realizations.append(realization)
        # print(realizationIndexes)
        print("readRealizations done")
        return realizations, realizationIndexes

        # raise NotImplementedError()

    def test(self, TestSamples, TestQueries, TestDecisions, predDecisions, n_jobs, logpath=None,
             preTrainPathResult=None):
        n_jobs = 1
        predLengths = []
        truelen = []

        approx_ratios = []
        approx_re_ratios = []

        len_ratios = []
        len_re_ratios = []

        testNum = len(TestSamples)
        if n_jobs == 1:
            #print("here")
            predLengths = self.test_block(predDecisions, 0)
        else:
            p = multiprocessing.Pool(n_jobs)
            block_size = int(testNum / n_jobs)
            Ys=p.starmap(self.test_block, ((predDecisions[i*block_size:min([len(TestQueries),(i+1)*block_size])], 1000) for i in range(n_jobs) ),chunksize=n_jobs)
            p.close()
            p.join()
            # decisions = []
            for len_block in Ys:
                predLengths.extend(len_block)

        for (query, sample, predlen) in zip(TestQueries, TestSamples, predLengths):
            truelen.append(sample.length)
            approx_ratios.append(predlen / sample.length)
            approx_re_ratios.append( sample.length / predlen )
            #print("{} {}".format(predlen, sample.length))

        mean_truelen = np.mean(np.array(truelen))
        std_truelen= np.std(np.array(truelen))

        mean_predLengths = np.mean(np.array(predLengths))
        std_predLengths = np.std(np.array(predLengths))

        mean_approx_ratios = np.mean(np.array(approx_ratios))
        std_approx_ratios = np.std(np.array(approx_ratios))

        mean_approx_re_ratios = np.mean(np.array(approx_re_ratios))
        std_approx_re_ratios = np.std(np.array(approx_re_ratios))


        len_ratio = mean_predLengths/mean_truelen
        len_re_ratio = mean_truelen / mean_predLengths
        #output = "truelen / predLengths: {} ({}) / {} ({})".format(Utils.formatFloat(mean_truelen), Utils.formatFloat(std_truelen),Utils.formatFloat(mean_predLengths), Utils.formatFloat(std_predLengths))
        #Utils.writeToFile(logpath, output, toconsole=True, preTrainPathResult=preTrainPathResult)
        #output = "predLengths: {} ({})".format(Utils.formatFloat(mean_predLengths), Utils.formatFloat(std_predLengths))
        #Utils.writeToFile(logpath, output, toconsole=True, preTrainPathResult=preTrainPathResult)
        #output = "approx_ratios, , approx_re_ratios, len_ratio, len_re_ratio"
        #Utils.writeToFile(logpath, output, toconsole=True, preTrainPathResult=preTrainPathResult)
        output = "{} ({}), {} ({}), {}, {}".format(Utils.formatFloat(mean_approx_ratios), Utils.formatFloat(std_approx_ratios), Utils.formatFloat(mean_approx_re_ratios), Utils.formatFloat(std_approx_re_ratios), Utils.formatFloat(len_ratio), Utils.formatFloat(len_re_ratio))
        Utils.writeToFile(logpath, output, toconsole=True, preTrainPathResult=preTrainPathResult)
        #output = "len_ratio: {}, len_re_ratio: {}".format(Utils.formatFloat(len_ratio), Utils.formatFloat(len_re_ratio))
        #Utils.writeToFile(logpath, output, toconsole=True, preTrainPathResult=preTrainPathResult)
        return len_ratio, len_re_ratio, mean_approx_ratios, std_approx_ratios, mean_approx_re_ratios, std_approx_re_ratios

    def test_block(self, Y_preds, nonsense):
        list_length = []
        for i in range(len(Y_preds)):
            length = 0
            for j in range((len(Y_preds[i]) // 2)):
                if Y_preds[i][2 * j] in self.stoGraph.nodes and Y_preds[i][2 * j + 1] in self.stoGraph.nodes[Y_preds[i][2 * j]].neighbor:
                    length += float(self.stoGraph.nodes[Y_preds[i][2 * j]].neighbor[Y_preds[i][2 * j + 1]][2])
                elif Y_preds[i][2 * j + 1] in self.stoGraph.nodes and Y_preds[i][2 * j] in self.stoGraph.nodes[Y_preds[i][2 * j+1]].neighbor:
                    length += float(self.stoGraph.nodes[Y_preds[i][2 * j + 1]].neighbor[Y_preds[i][2 * j]][2])
                else:
                    print("edge not existing")
                    print(Y_preds)
                    return None
            list_length.append(length)
        return list_length

def main():
    pass
    '''
    dataname = "kro"
    stoGraphPath = "data/{}/{}_model".format(dataname, dataname)
    vNum = 1024
    stoGraph = StoGraph(stoGraphPath, vNum)
    st_usco = Steiner_USCO(stoGraph)
    x_scale = 10
    num = 10
    sample_type = "normal"
    Wpath = "data/{}/{}_DC_samples_{}_{}_{}_{}".format(dataname, dataname, x_scale, 10, sample_type, num)
    st_usco.genSamples(Wpath=Wpath, scale_x=x_scale, scale_y=10, num = 2, thread = 2)
    '''

# g = Graph(9)
if __name__ == "__main__":
    # pass
    main()
