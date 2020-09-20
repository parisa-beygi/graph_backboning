import csv
import numpy as np
import matplotlib.pyplot as plt
from backbone_extraction import *
from nullmodels.wrg import WRG

def plot_single_edges_fraction_vs_significance_level(filtering, graph_filename):
    """Read the pvalues file which each line as: u, v, p_value where p_value corresponds to the edge (u,v)"""
    filter_name = filtering.get_filter_name()
    pvalues_filename = '../results/{}_pvalues.csv'.format(filter_name)
    edges = []
    pvalues = []
    with open(pvalues_filename, newline='') as csvfile:
        edge_pvalue_list = list(csv.reader(csvfile))

        for edge_pvalue in edge_pvalue_list:
            u, v = filtering.read_pvaluefile_edges(edge_pvalue[0], edge_pvalue[1])
            pvalue = float(edge_pvalue[2])
            edges.append((u,v))
            pvalues.append(pvalue)

    print ('pvalues read')
    print (edges, pvalues)
    graph = filtering.read_graph(graph_filename)
    E0 = filtering.get_num_of_edges(graph)
    fractions = []
    significance_level_range = np.linspace(1e-4, 1e4, 1000)
    m_significance_level_range = list(map(lambda x:x/filtering.get_num_statistical_tests(edges), significance_level_range))
    x_ticks = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    for alpha in significance_level_range:
            backbone = filtering.filter(graph.copy(), edges, pvalues, alpha)
            Eb = filtering.get_num_of_edges(backbone)
            fractions.append(Eb/E0)

    # plt.plot(significance_level_range, fractions)
    # plt.xlabel('significance level')
    # plt.ylabel('EB/E0')
    # plt.xticks(x_ticks)
    # plt.show()
    # plt.savefig('../results/mlf_edges_fraction_alpha.png')
    # mesal = [i for i in range(1000)]
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.plot(m_significance_level_range, fractions, color ="green")
    # ax.plot(m_significance_level_range, mesal, color= "blue")
    # ax.set_xlim(1e-8,1.1e0)
    plt.xlabel('significance level')
    plt.ylabel('EB/E0')
    plt.savefig('../results/{}_edges_fraction_alpha.png'.format(filter_name))


def plot_edges_fraction_vs_significance_level(*filterings, graph_filename):
    """Read the pvalues file which each line as: u, v, p_value where p_value corresponds to the edge (u,v)"""
    filtering_edges = []
    filtering_pvalues = []
    for filtering in filterings:
        filter_name = filtering.get_filter_name()
        pvalues_filename = '../results/{}_pvalues.csv'.format(filter_name)
        edges = []
        pvalues = []
        with open(pvalues_filename, newline='') as csvfile:
            edge_pvalue_list = list(csv.reader(csvfile))

            for edge_pvalue in edge_pvalue_list:
                u, v = filtering.read_pvaluefile_edges(edge_pvalue[0], edge_pvalue[1])
                pvalue = float(edge_pvalue[2])
                edges.append((u,v))
                pvalues.append(pvalue)

        filtering_edges.append(edges)
        filtering_pvalues.append(pvalues)

    print ('pvalues read')
    # print (edges, pvalues)

    fig, ax = plt.subplots()
    ax.set_xscale("log")

    for filtering, edges, pvalues in zip(filterings, filtering_edges, filtering_pvalues):

        graph = filtering.read_graph(graph_filename)
        E0 = filtering.get_num_of_edges(graph)
        fractions = []
        correction_term = filtering.get_num_statistical_tests(edges)
        significance_level_range = np.linspace(correction_term*1e-3, correction_term, 1000)
        m_significance_level_range = list(map(lambda x:x/correction_term, significance_level_range))
        # x_ticks = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
        for alpha in significance_level_range:
                backbone = filtering.filter(graph.copy(), edges, pvalues, alpha)
                Eb = filtering.get_num_of_edges(backbone)
                fractions.append(Eb/E0)

        ax.plot(m_significance_level_range, fractions)

    plt.legend([filtering.get_filter_name() for filtering in filterings])

    plt.xlabel('significance level')
    plt.ylabel('EB/E0')
    plt.savefig('../results/all_filters_edges_fraction_alpha.png')


if __name__ == "__main__":
    # mlf = MLFiltering()
    # fil = WeightedMotifFiltering()
    plot_edges_fraction_vs_significance_level(MLFiltering(), PolyaFiltering(0.2, 10.0, 1), PolyaFiltering(7.0, 10.0, 1), WeightedMotifFiltering(50), graph_filename = '../../datasets/vnet_hyperego.graphml')





