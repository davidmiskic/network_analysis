import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
import copy
import random
import math
import sys
from statistics import mean
from helper import *

"""
Compute degree distributions, check if they follow power law
"""
def degree_dist():
    graphjava = nx.read_pajek("java.net")
    graphlucene = nx.read_pajek("lucene.net")
    print(nx.info(graphjava))
    print(nx.is_directed(graphjava))
    print(nx.info(graphlucene))
    print(nx.is_directed(graphlucene))

    # Lucene in degree, Java in degree
    # degin,degout = [graph.in_degree(n) for n in graph.nodes()], [graph.out_degree(n) for n in graph.nodes()]
    def compute_gamma(degrees, kmin):
        sum = 0.0
        ndot = 0
        for deg in degrees:
            if deg >= kmin:
                ndot += 1
                sum += math.log(deg/(kmin-0.5))
        return 1 + ndot*math.pow(sum, -1)

    degjava = [graphjava.in_degree(n) for n in graphjava.nodes()]
    deglucene = [graphlucene.in_degree(n) for n in graphlucene.nodes()]

    test_plot_gamma("Java in", compute_gamma(degjava, 10), degjava, graphjava, 10)
    test_plot_gamma("Lucene in", compute_gamma(deglucene, 15), deglucene, graphlucene, 15)


"""
Analyze vulnerabilty of internet when nodes with largest degrees are removed
"""
def internet_removal():
    graphinternet = nx.read_pajek("nec.net")
    graphinternet = graphinternet.to_undirected()

    # plot the fraction of nodes in LCC after removing 0%, 10%, 20%, 30%, 40% and 50%

    percs = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    datarand = [random_removal(graphinternet, share) for share in [0, 0.1, 0.2, 0.3, 0.4, 0.5]]
    datarand = [datarand[i]/(len(graphinternet)*(1-percs[i])) for i in range(len(percs))]

    datahub = [hub_removal(graphinternet, share) for share in [0, 0.1, 0.2, 0.3, 0.4, 0.5]]
    datahub = [datahub[i]/(len(graphinternet)*(1-percs[i])) for i in range(len(percs))]

    def plotLCC(data, type):
        percentages = [0, 10, 20, 30, 40, 50]
        plt.figure()
        plt.plot(percentages, data, "ro")
        plt.xlabel("%")
        plt.title("share of nodes in LCC after " + type + " removal")
        plt.show()

    plotLCC(datarand, "random")
    plotLCC(datahub, "hub")

    # Finally, repeat the experiments also for a random graph

    randgraph = nx.generators.random_graphs.gnm_random_graph(75885, 357317)

    datarandgraphrand = [random_removal(randgraph, share) for share in [0, 0.1, 0.2, 0.3, 0.4, 0.5]]
    datarandgraphrand = [datarandgraphrand[i] / (len(randgraph) * (1 - percs[i])) for i in range(len(percs))]
    datarandgraphhub = [hub_removal(randgraph, share) for share in [0, 0.1, 0.2, 0.3, 0.4, 0.5]]
    datarandgraphhub = [datarandgraphhub[i] / (len(randgraph) * (1 - percs[i])) for i in range(len(percs))]
    plotLCC(datarandgraphrand, "random-RG")
    plotLCC(datarandgraphhub, "hub-RG")


"""
Analyze social network graph
"""
def social_network():
    graph = nx.read_adjlist("social.adj")
    print(nx.info(graph))

    sequence = init_random_walk(graph)
    induced = graph.subgraph(sequence)


    sigmainduced = nx.algorithms.smallworld.sigma(induced, niter=25, nrand=10)
    print(sigmainduced)
    omegainduced = nx.algorithms.smallworld.omega(induced)
    print(omegainduced)

    sigmaoriginal = nx.algorithms.smallworld.sigma(graph, niter=25, nrand=10)
    print(sigmaoriginal)
    omegaoriginal = nx.algorithms.smallworld.omega(graph)
    print(omegaoriginal)

    print("calculating lengths for induced")
    len_induced = nx.algorithms.shortest_paths.generic.average_shortest_path_length(induced)
    print(len_induced, math.log(len_induced)) # 5.532984776526998 1.7107274125163803
    clust_induced = nx.algorithms.cluster.average_clustering(induced)
    print("cci", clust_induced) # 0.4710909181675598

    print("calculating lengths for original")
    len_graph = nx.algorithms.shortest_paths.generic.average_shortest_path_length(graph)
    print(len_graph, math.log(len_graph)) # 7.4855400514784 2.01297316643494
    clust_graph = nx.algorithms.cluster.average_clustering(graph)
    print("cco", clust_graph) #  0.26594522430104395

    print(nx.info(induced))
    deggraph = [graph.degree(n) for n in graph.nodes()]
    print(mean(deggraph), math.log(sum(deggraph)/len(deggraph)))
    test_plot_gamma("Social network", compute_gamma(deggraph, 3), deggraph, graph, 3)
    deginduced = [induced.degree(n) for n in induced.nodes()]
    print(mean(deginduced), math.log(sum(deginduced) / len(deginduced)))
    test_plot_gamma("Induced network", compute_gamma(deginduced, 3), deginduced, induced, 3)
    plotdeg(graph, "social network", "r")
    plotdeg(induced, "induced network", "b")