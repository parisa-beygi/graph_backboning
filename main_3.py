from backbone_extraction import *
import numpy as np
from preprocess import read_egos
import networkx as nx
import igraph as ig
from igraph import *
import os

if __name__ == "__main__":
    # print('Main')
    # print ('get_ego')
    # ego_filename = '../../datasets/ego_1.graphml'
    # if not os.path.exists(ego_filename):
    #     graph_filename = "../../datasets/vnet.graphml"
    #     dataset = GraphDataset(graph_filename)
    #     ego = nx.ego_graph(dataset.G, 'n1')
    #     nx.write_graphml(ego, ego_filename)
    # else:
    #     ego = nx.read_graphml(ego_filename)
    #
    # print ('done_ego')
    # print(len(list(ego.nodes())), len(list(ego.edges())))



    # ego_filename = '../../datasets/vnet_hyperego.graphml'
    #
    # mlf = MLFiltering()
    # ig_ego = mlf.read_graph(ego_filename)
    # print ('#E = ', len(ig_ego.es))
    # print ('computing mlf pvalues')
    # edges, pvalues = mlf.compute_pvalues(ig_ego)
    # mlf.save_pvalues(edges, pvalues, '../results')
    # print ('ml filtering')
    # bb = mlf.filter(ig_ego, edges, pvalues, 0.05)
    # print ('#Eb = ', len(bb.es))


    ego_filename = '../../datasets/vnet_hyperego.graphml'

    wmf = WeightedMotifFiltering(50)
    graph = wmf.read_graph(ego_filename)
    wmf.initilize(graph)
    print ('#E = ', len(graph.edges))
    print ('computing wmf pvalues')
    edges, pvalues = wmf.compute_pvalues(graph)
    wmf.save_pvalues(edges, pvalues, '../results')
    # print ('wmf filtering')
    # bb = wmf.filter(graph, edges, pvalues, 0.05)
    # print ('#Eb = ', len(bb.edges))


    # pf = PolyaFiltering(7.0, 10.0, 1)
    # gego = pf.read_graph(ego_filename)
    # print ('graph dimensions: ', len(gego.nodes()), len(gego.edges()))
    # print ('computing pf pvalues')
    # edges, pvalues = pf.compute_pvalues(gego)
    # pf.save_pvalues(edges, pvalues, '../results')
    # print ('polya filtering')
    # bb = pf.filter(gego, edges, pvalues, 0.05)
    # # print (list(bb.edges()))
    # print('#Eb = ', len(bb.edges()))


    # print(type(dataset.W))
    # print(dataset.W.toarray())
    # print(type(dataset.W.toarray()))
    # print(len(list(dataset.G.nodes())))

    # ego, egonet = read_egos.get_largest_ego(dataset.G)
    # print (nx.to_numpy_matrix(egonet))

    # node = 0
    # ego = None
    # for n, egonet in read_egos.get_egos(graph_filename):
    #     print (n)
    #     if len(list(egonet.nodes())) > 1000:
    #         node = n
    #         ego = egonet
    #         break
    # print (node, len(list(ego.edges())))

    # W = np.array([[0, 2, 3], [4, 0, 6], []])
    # pf = PolyaFiltering(1, 0, 1)
    # pf.filter(W)


    # Test Dianati filter
    # ego = Graph()
    # ego.add_vertices(3)
    # ego.add_edges([(0, 1), (1, 2), (2, 0)])
    # ego.es['weight'] = [2, 3, 1]
    #
    # # ego.add_edges([('0', '1', 2), ('1', '2', 3), ('2', '0', 1)])
    # from baselines import MLF_filter
    # print ('Started compute_significance')
    # MLF_filter.compute_significance(ego)
    # print ('Started MLF pruning')
    # backbone = MLF_filter.prune(ego, field='significance', percent=None, num_remove=None, significance_level=0.5)
    # print (len(list(backbone.vs)), len(list(backbone.es)))

