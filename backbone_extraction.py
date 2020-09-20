from baselines import MLF_filter
import networkx as nx
import igraph as ig
from math import *
import numpy as np
# import matlab.engine
import os
from utils import sum_combinations
from utils import math_computations
import math
from collections import Counter
import json
import gc
import time
import random
import matplotlib.pyplot as plt

class Filtering():
    """Class for significance testing backbone extraction methods"""
    def __init__(self):
        pass

    def read_graph(self, graph_filename):
        return nx.read_graphml(graph_filename)

    def write_graph(self, G, dest_filename):
        nx.write_graphml(G, dest_filename)

    def compute_pvalues(self, G):
        pass

    def get_num_statistical_tests(self, edges_list):
        return len(edges_list)

    def filter(self, G, edges, pvalues, significance_level):
        # multivariate_significance_level = significance_level
        multivariate_significance_level = significance_level/self.get_num_statistical_tests(edges)
        print ("pvalues: ", pvalues)
        deathrow = [edges[i] for i in range(len(edges)) if not self.is_significant(pvalues[i], multivariate_significance_level)]
        return self.prune(G, deathrow)

    def is_significant(self, pvalue, threshold):
        return (pvalue < threshold)

    def prune(self, G, edges):
        G.remove_edges_from(edges)
        return G

    def save_pvalues(self, edge_list, p_values, save_dir):
        # print ('edge, pvalue')
        # for i in zip(edge_list, p_values):
        #     print (i)
        rows = [[e[0], e[1], p] for (e, p) in zip(edge_list, p_values)]
        # print (rows)
        l = np.array(rows, dtype=str)
        print(l)
        np.savetxt(os.path.join(save_dir, '{}_pvalues.csv'.format(self.get_filter_name())), l, delimiter=',', fmt='%s')
        print ('num of distinct pvalues: ', len(set(p_values)))
        plt.hist(p_values, bins=len(set(p_values)))
        plt.ylabel('Probability')
        plt.xlabel('p_value');
        plt.savefig('../results/hist_pvalues_{}.png'.format(self.get_filter_name()))

    def get_num_of_edges(self, graph):
        return len(graph.edges())

    def read_pvaluefile_edges(self, v1, v2):
        return v1, v2

class WeightedMotifFiltering(Filtering):

    def __init__(self, deg):
        super().__init__()
        self.null_degree = deg

    def get_nullmodel_degree(self):
        return self.null_degree

    def initilize(self, graph):
        """Initialize the parameters of the WRG (Weighted Random Graph)."""
        # self.graph = graph
        self.N = len(graph.nodes())

        self.set_weight_list_and_max_weight(graph)
        print('self.max_weight : ', self.max_weight)

        self.set_max_degree(graph)
        print('self.max_degree : ', self.max_degree)

        self.valid_num_set, self.valid_pair_set = self.get_valid_num_set(self.max_weight)

        self.max_total_W_motif = self.max_weight * self.max_weight * self.max_degree
        print('self.max_total_W_motif : ', self.max_total_W_motif)

        W = self.get_weights_sum()
        self.p = 2 * W / (self.N * (self.N - 1) + 2 * W)

        self.zero_W_motif = 1 - pow(self.p, 2)
        self.nonzero_W_motif_base = pow((1 - self.p), 2)

        self.single_motif_dict = {}
        self.total_motif_dict = {}
        self.cmb = sum_combinations.Combinations(self.null_degree, self.max_weight * self.max_weight, self.valid_num_set)
        self.load_combinations_dict()

    def get_valid_num_set(self, n):
        s = set()
        s_pairs = set()

        for i in range(1, n + 1):
            for j in range(i, n + 1):
                s.add(i * j)
                s_pairs.add((i, j))

        return s, s_pairs

    def load_combinations_dict(self):
        dict_path = "../data/combination_dict_{}_{}_thth.json".format(self.get_nullmodel_degree(),9)
        print('Loading combination ({}, {}) dictionary'.format(self.get_nullmodel_degree(), 9))
        if not os.path.exists(dict_path):
            print('File not found! Started computing dictionary ... ')

            for chunk_combs in self.cmb.find_combinations_dp(self.max_total_W_motif):
                del chunk_combs
                gc.collect()

            with open(dict_path, 'w') as fp:
                json.dump(self.cmb.get_dict(), fp, sort_keys=True)

        else:
            print ('loading ...')
            f = open(dict_path, 'r')
            temp_dict = json.load(f)

            self.cmb.set_dict({int(k): v for k, v in temp_dict.items()})
            print ('loaded!')


    def save_combinations_dict(self, new_dict):
        fw = open("../data/combination_dict.json", 'w')
        json.dump(new_dict, fw)

    @classmethod
    def from_adjmatrix(cls, adj_mat):
        return cls(nx.from_numpy_matrix(adj_mat))

    def set_weight_list_and_max_weight(self, graph):
        raw_weight_list = list(graph.edges.data('weight'))
        self.weighted_edges_list = raw_weight_list
        self.max_weight = max(raw_weight_list, key=lambda x: x[2])[2]

        if self.max_weight > 3:
            print('raw max weight is ', self.max_weight)
            k = self.max_weight ** (1 / 3)
            self.max_weight = 3
            self.weighted_edges_list = list(map(lambda x: (x[0], x[1], round(math.log(x[2], k))), raw_weight_list))

        # self.edge_weight_dict = dict({(u, v): w for (u, v, w) in self.weighted_edges_list})
        self.edge_weight_dict = {}
        for (u, v, w) in self.weighted_edges_list:
            self.edge_weight_dict[(u, v)] = self.edge_weight_dict[(v, u)] = w
        # print ('EDGE WEIGHT DICT VALUES: ', set(self.edge_weight_dict.values()))

    def set_max_degree(self, graph):
        self.max_degree = max(list(map(lambda x: graph.degree[x], list(graph.nodes()))))

    def get_weights_sum(self):
        s = 0
        for (u, v, weight) in self.weighted_edges_list:
            s += weight
        return s

    def get_edge_prob(self, w):
        return pow(self.p, w) * (1 - self.p)

    def get_single_nonzero_W_motif_probability(self, a, b):
        return pow(self.p, a + b)

    def get_nonzero_W_motif(self, k):
        total = 0
        div_pairs = math_computations.get_divisor_pairs(k)

        pairs = self.valid_pair_set.intersection(set(div_pairs))
        print ('pairs: ', pairs)
        for (a, b) in pairs:
            # if a <= self.max_weight and b <= self.max_weight:
            total += self.get_single_nonzero_W_motif_probability(a, b)

        return self.nonzero_W_motif_base * total

    def W_motif_comb(self, combination):
        # set_trace()
        total_mult = 1
        for i in combination:
            if i not in self.single_motif_dict:
                self.single_motif_dict[i] = self.get_nonzero_W_motif(i)
            total_mult = total_mult * self.single_motif_dict[i]

        total_mult = total_mult * pow(self.get_single_motif_zero_probability(), self.null_degree - len(combination))
        all_permutations = math.factorial(self.null_degree)
        zero_permutations = math.factorial(self.null_degree - len(combination))
        nonzero_permutations = 1
        counts = Counter(combination)
        for k in counts:
            nonzero_permutations = nonzero_permutations * math.factorial(counts[k])

        d = all_permutations / (zero_permutations * nonzero_permutations)
        # print ("total_mult: ", total_mult)
        # print ("num of digits of d: ", int(math.log10(d))+1)

        # return total_mult
        return d * total_mult


    def get_single_motif_zero_probability(self):
        return self.zero_W_motif

    def get_total_W_motif(self, n):
        if n == 0:
            return pow(self.get_single_motif_zero_probability(), self.null_degree)


        total_sum = 0

        print('hit sum = {}'.format(n))
        # n = 90 if n > 90 else n
        all_combinations = self.cmb.get_dict()[n]

        res_list = list(map(self.W_motif_comb, all_combinations))
        total_sum += sum(res_list)

        return total_sum

    # def get_pvalue_term(self, w):
    #     if w not in self.total_motif_dict:
    #         self.total_motif_dict[w] = self.get_total_W_motif(w)
    #
    #     return self.total_motif_dict[w]

    def get_p_value(self, W_star):
        # print ('*******Started calculating pvalue*******')

        total = 0
        print ('W_star: ', W_star)
        for w in range(0, W_star + 1):
            # print ('In get_p_value() >>> W = ', w)
            if w not in self.total_motif_dict:
                self.total_motif_dict[w] = self.get_total_W_motif(w)
            print ('in get_pvalues(): ', w, self.total_motif_dict[w])
            total += self.total_motif_dict[w]
        # print ('total: ', total)
        return (1 - total)

    def compute_pvalues(self, graph):
        p_values = np.array([])
        edge_list = []
        print ('started computing pvalues')
        for (u, v) in graph.edges():
            print (u, v)
            total_W_motif = 0
            common_neighbors = list(nx.common_neighbors(graph, u, v))
            common_neighbors = common_neighbors if len(common_neighbors) <= self.null_degree else random.sample(common_neighbors, k = self.null_degree)

            for c in common_neighbors:
                W_motif = self.edge_weight_dict[(u, c)] * self.edge_weight_dict[(v, c)]
                print ('W_motif: ', W_motif)
                # if total_W_motif + W_motif > self.null_degree * self.max_weight * self.max_weight:
                #     break
                total_W_motif += W_motif

            print (type(total_W_motif))
            print (total_W_motif)
            pval = self.get_p_value(total_W_motif)
            # print ('pval: ', pval)
            # print ('In main >>> p_value({},{}) = {} in {} seconds.\n'.format(u,v, pval, time.time() - start_time))

            p_values = np.append(p_values, pval)
            edge_list.append((u, v))

        # print (p_values)
        # print (len(p_values))

        return edge_list, p_values

    def get_filter_name(self):
        return 'wm_{}_3'.format(self.get_nullmodel_degree())

class MaxWeightedMotifFiltering(WeightedMotifFiltering):
    def initilize(self, graph):
        """Initialize the parameters of the WRG (Weighted Random Graph)."""
        # self.graph = graph
        self.N = len(graph.nodes())

        self.set_weight_list_and_max_weight(graph, 3)
        print('self.max_weight : ', self.max_weight)

        self.set_max_degree(graph)
        print('self.max_degree : ', self.max_degree)

        self.valid_num_set, self.valid_pair_set = self.get_valid_num_set(self.max_weight)

        W = self.get_weights_sum()
        self.p = 2 * W / (self.N * (self.N - 1) + 2 * W)

        self.zero_W_motif = 1 - pow(self.p, 2)
        self.nonzero_W_motif_base = pow((1 - self.p), 2)

        self.total_motif_dict = {}

    def set_weight_list_and_max_weight(self, graph, a):
        raw_weight_list = list(map(lambda x: (x[0], x[1], int(x[2])), list(graph.edges.data('weight'))))
        self.weighted_edges_list = raw_weight_list
        self.max_weight = max(raw_weight_list, key=lambda x: x[2])[2]

        if self.max_weight > a:
            print('raw max weight is ', self.max_weight)
            k = self.max_weight ** (1 / a)
            self.max_weight = a
            self.weighted_edges_list = list(map(lambda x: (x[0], x[1], round(math.log(x[2], k))), raw_weight_list))

        # self.edge_weight_dict = dict({(u, v): w for (u, v, w) in self.weighted_edges_list})
        self.edge_weight_dict = {}
        for (u, v, w) in self.weighted_edges_list:
            self.edge_weight_dict[(u, v)] = self.edge_weight_dict[(v, u)] = w
        # print ('EDGE WEIGHT DICT VALUES: ', set(self.edge_weight_dict.values()))

    def get_nonzero_W_motif(self, k):
        total = 0
        div_pairs = math_computations.get_divisor_pairs(k)
        # if k ==11:
        #     print ("div_pairs ", div_pairs)
        #     print ("self.valid_pair_set ", self.valid_pair_set)

        # pairs = self.valid_pair_set.intersection(set(div_pairs))
        # print ('pairs: ', pairs)
        for (a, b) in div_pairs:
            # if a <= self.max_weight and b <= self.max_weight:
            total += pow(self.p, a + b)

        return self.nonzero_W_motif_base * total

    def compute_pvalues(self, graph):
        p_values = np.array([])
        edge_list = []
        for (u, v) in graph.edges():
            max_weighted_w_motif = 0
            for c in nx.common_neighbors(graph, u, v):
                max_weighted_w_motif = max(self.edge_weight_dict[(u, c)] * self.edge_weight_dict[(v, c)], max_weighted_w_motif)
            print ('{}, {} max_weighted_w_motif'.format(u, v), max_weighted_w_motif)
            pval = self.get_p_value(max_weighted_w_motif)
            print ('{}, {} pvalue'.format(u, v), pval)
            p_values = np.append(p_values, pval)
            edge_list.append((u, v))
            # time.sleep(100)
        return edge_list, p_values

    def get_total_W_motif(self, n):
        # print ('get_total_Wmotif: ', n)
        if n == 0:
            # print ('here: ', pow(self.zero_W_motif, self.N - 2))
            return pow(self.zero_W_motif, self.max_degree - 2)
            # return -log(pow(self.zero_W_motif, self.N - 2))


        p_max_motif = self.get_nonzero_W_motif(n)
        # print ("p_max_motif: ", p_max_motif)
        # print("log:", log(p_max_motif))
        p_other_motifs = 0
        for m in range(n):
            mm = self.get_nonzero_W_motif(m) if m else pow(self.zero_W_motif, self.max_degree - 2)
            # print ('mm: ', mm)
            p_other_motifs += mm

        # print ('here: ', (self.N - 2) * p_max_motif * pow(p_other_motifs, self.N - 3))
        # print ('but')
        # print (self.N - 2)
        print ('***get_total_W_motif***')
        print (p_max_motif)
        print (p_other_motifs)
        print (pow(p_other_motifs, self.max_degree - 3))
        return (self.max_degree - 2) * p_max_motif * pow(p_other_motifs, self.max_degree - 3)
        # return (-log(self.N - 2) - log(p_max_motif) -(self.N - 3)*log(p_other_motifs))
        # return -log((self.N - 2) * p_max_motif * pow(p_other_motifs, self.N - 3))
        # return (self.N - 2) * p_max_motif

    # def is_significant(self, pvalue, threshold):
    #     return (pvalue > -log(threshold))

    def get_filter_name(self):
        return 'maxwm'

class PolyaFiltering(Filtering):
    """The Polya Filter which assesses the significance of links given a null hypothesis based on the Polya problem"""

    def __init__(self, a, apr_lvl, parallel):
        super().__init__()
        self.a = a
        self.apr_lvl = apr_lvl
        self.parallel = parallel

    def get_num_statistical_tests(self, edges_list):
        return 2*len(edges_list)

    def compute_pvalues(self, G):
        eng = matlab.engine.start_matlab()
        # print ('G.edges(): ', list(G.edges.data('weight')))
        W = nx.to_numpy_matrix(G)
        nodes_list = list(G.nodes())
        edges = [(nodes_list[x[0]], nodes_list[x[1]]) for x in np.transpose(np.nonzero(np.triu(W)))]
        # print('edges: ', edges)
        W = W.tolist()
        # print ('W: ', W)
        mat_W = matlab.double(W)
        # print ('mat_W: ', mat_W)
        pvalues = eng.PF(mat_W, self.a, self.apr_lvl, self.parallel)
        pvalues = [p[0] for p in pvalues]
        # print ('pvalues: ', type(pvalues), pvalues)
        return edges, pvalues

    def get_filter_name(self):
        return 'polya_{}'.format(self.a)

class MLFiltering(Filtering):
    """The Marginal Likelihood Filter which is based on a null hypothesis of max entropy given the sequence of node strengths"""
    def __init__(self):
        super().__init__()

    def read_graph(self, graph_filename):
        return ig.read(graph_filename,format="graphml")

    def write_graph(self, G, dest_filename):
        ig.write(G, dest_filename, format="graphml")

    def get_edges_pvalues(self, G, field='significance'):
        edges = []
        pvalues = []
        for e in G.es:
            # print ('******************************')
            # print ((type(G.vs[e.source]), type(G.vs[e.target])))
            # print ((G.vs[e.source].index, G.vs[e.target].index))

            edges.append((e.source, e.target))
            pvalues.append(e[field])
        return edges, pvalues

    def compute_pvalues(self, G):
        print('Started compute_significance')
        MLF_filter.compute_significance(G)

        edges, pvalues = self.get_edges_pvalues(G)
        return edges, pvalues

    def is_significant(self, pvalue, threshold):
        return (pvalue > -log(threshold))

    def prune(self, G, edges):
        # print ('in prune')
        # print (edges)
        G.delete_edges(edges)
        return G

    def get_filter_name(self):
        return 'mlf'

    def get_num_of_edges(self, graph):
        return len(graph.es)

    def read_pvaluefile_edges(self, v1, v2):
        return int(v1), int(v2)

class TestEdgesVSSignificanceLevel():
    def __init__(self, graph_dataset, filtering_method):
        self.graph_dataset = graph_dataset
        self.filtering_method = filtering_method

    def plot(self):
        pass


