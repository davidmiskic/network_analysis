import networkx as nx
import random
import sys
import json
import math
import scipy.stats
import numpy
import matplotlib.pyplot as plt
sys.setrecursionlimit(1500)


"""
Strongly Connected Component
"""
i = 0
sccs = []
sccsi = 0
isccs = {}

# iterative Tarjan algorithm
def strongConnections(graph, node, edgeindexes, edgelowlink, edgeMember, stack):
    global i, sccsi, isccs, sccs
    print(i,)
    visited = [0 for i in range(0, len(graph))]
    recIt = [[node, 0]]

    while len(recIt) > 0:
        current = recIt.pop()
        if current[1] == 0:
            edgeindexes[current[0]] = i
            edgelowlink[current[0]] = i
            i += i
            stack.append(current[0])
            edgeMember[current[0]] = True
        for k in graph[current[0]]:
            if edgeindexes[k] == -1:
                recIt.append([current[0], k])
                recIt.append([k, 0])
                break
            elif edgeMember[k]:
                edgelowlink[current[0]] = min(edgelowlink[current[0]], edgeindexes[k])

        if edgeindexes[k] == -1:
            continue
        if edgeindexes[current[0]] == edgelowlink[current[0]]:
            group = []
            out = stack.pop()
            while current[0] != out:
                edgeMember[out] = False
                group.append(out)
                isccs[out] = sccsi
                out = stack.pop()
            sccs.append(group)
            sccsi += 1
        if len(recIt)>0:
            k = current[0]
            current = recIt.pop()
            edgelowlink[current[0]] = min(edgelowlink[current[0]], edgelowlink[k])

# compute strongly connected components
def dirStronglyConnectedComponents(g):
    nnodes = len(nx.nodes(g)) + 1
    edgeindexes = [-1 for e in range(0, nnodes)]
    edgelowlink = [-1 for e in range(0, nnodes)]
    edgeMember = [False for e in range(0, nnodes)]
    stack = []
    for node in g:
        if edgeindexes[node] == -1: strongConnections(g, node, edgeindexes, edgelowlink, edgeMember, stack)


"""
Diameter
"""
def floydWarshall(graph):
    size = max(graph.nodes) + 1
    dist = numpy.ones((size, size), dtype=numpy.int32)*100000
    for edge in list(graph.edges):
        dist[edge[0]][edge[1]] = 1
    for node in list(graph.nodes):
        dist[node][node] = 0
    nodes = list(graph.nodes)
    for k in range(0, len(graph)):
        if k in nodes:
            print(k, end="")
            for i in range(0, len(graph)):
                for j in range(0, len(graph)):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

    return dist


"""
Correlation between different features and importance in highway network
"""
def position():
    highways = nx.read_pajek("highways.net")
    highways = nx.Graph(highways)
    print(nx.info(highways))
    degrees = nx.degree(highways)
    degrees = [deg[1] for deg in degrees]
    clust = nx.clustering(highways)
    clust = [clust[key] for key in clust.keys()]
    harmcent = nx.harmonic_centrality(highways)
    harmcentn = 1/len(highways.nodes)
    harmcent = [harmcent[key]*harmcentn for key in harmcent.keys()]
    #print(degrees)
    #print(clust)
    #print(harmcent)

    f = open("highways.net", "r")
    f.readline()
    line = f.readline()
    loads = []
    while "*edges 125" not in line:
        load = line.split(" ")[-1].replace("\n", "")
        loads.append(float(load))
        line = f.readline()
    coeffdeg, pval = scipy.stats.spearmanr(loads, degrees)
    coeffclust, pval = scipy.stats.spearmanr(loads, clust)
    coeffharm, pval = scipy.stats.spearmanr(loads, harmcent)

    dloads = {}
    nodes = list(nx.nodes(highways))
    for i in range(0, len(loads)):
        dloads[nodes[i]] = [loads[i], harmcent[i]]

    sortedharm = sorted(harmcent, reverse=True)
    #print(coeffdeg, coeffclust, coeffharm)

    #sortedharm = sorted(degrees, reverse=True)
    #print(sortedharm)
    print(coeffdeg, coeffclust, coeffharm)

    for i in range(0, 10):
        val = sortedharm[i]
        for key in dloads.keys():

            if dloads[key][1] == val:
                print(key, dloads[key])
                dloads[key][1] = -1
                break