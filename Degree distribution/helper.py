import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
import copy
import random
import math
import sys
from statistics import mean

def test_plot_gamma(name, gamma, degrees, graph, kmin):
    degprob = dict(Counter(degrees))
    degprob.pop(0)
    for key in degprob.keys(): degprob[key] = math.pow(key, -gamma)
    # deggama = [math.pow(n, -gamma) for n in degprob]

    degprob2 = dict(Counter(degrees))
    for key in degprob2: degprob2[key] = degprob2[key] / len(graph)
    if 0 in degprob2.keys(): degprob2.pop(0)
    mintoplot = min(degprob2.values())
    topop = [key for key in degprob.keys() if degprob[key] < mintoplot]
    for key in topop: degprob.pop(key)

    plt.figure()
    plt.loglog(degprob.keys(), degprob.values(), "bo", label="power law y: " + str(round(gamma, 5)))
    plt.loglog(degprob2.keys(), degprob2.values(), "go", label="in-degree")
    plt.xlabel("k")
    plt.ylabel("Pk")
    plt.legend()
    plt.title(name + " degree probability distribution and kmin: " + str(kmin))
    plt.show()


def plot3deg(graph, name, colorchar):
    deg = [graph.degree(n) for n in graph.nodes()]
    degprob = dict(Counter(deg))
    for key in degprob:
        degprob[key] = degprob[key] / len(graph)
    degin,degout = [graph.in_degree(n) for n in graph.nodes()], [graph.out_degree(n) for n in graph.nodes()]
    deginprob, degoutprob = dict(Counter(degin)), dict(Counter(degout))
    print(deginprob, degoutprob)
    if 0 in degprob.keys(): degprob.pop(0)
    if 0 in degoutprob.keys(): degoutprob.pop(0)
    if 0 in deginprob.keys(): deginprob.pop(0)
    for key in deginprob: deginprob[key] = deginprob[key] / len(graph)
    for key in degoutprob: degoutprob[key] = degoutprob[key] / len(graph)
    print(sum(degprob.values()), sum(deginprob.values()), sum(degoutprob.values()))


    plt.figure()
    plt.loglog(degprob.keys(), degprob.values(), colorchar + "o", label="degree")
    plt.loglog(deginprob.keys(), deginprob.values(), "bo", label="in-degree")
    plt.loglog(degoutprob.keys(), degoutprob.values(), "go", label="out-degree")
    plt.legend()
    plt.xlabel("k")
    plt.ylabel("Pk")
    plt.title(name + " degree probability distribution")
    plt.show()


def random_removal(graph, howmuch):
    workingGraph = copy.deepcopy(graph)
    nnodes = len(workingGraph)
    ntoremove = math.ceil(nnodes * howmuch)
    print(nnodes, ntoremove)
    nodes = list(workingGraph.nodes())
    toRemove = random.sample(nodes, ntoremove)
    for i in toRemove:
        workingGraph.remove_node(i)
    print("removed", nnodes - len(workingGraph), "goal", ntoremove)
    largest_cc = max(nx.connected_components(workingGraph), key=len)
    print(len(largest_cc))
    return len(largest_cc)

def hub_removal(graph, howmuch):
    workingGraph = copy.deepcopy(graph)
    deg = sorted(list(workingGraph.degree), key=lambda tup: tup[1], reverse=True)
    nnodes = len(workingGraph)
    ntoremove = math.ceil(nnodes * howmuch)
    print("hub removal", nnodes, ntoremove)
    for i in range(0, ntoremove):
        if i % 10 == 0:
            deg = sorted(list(workingGraph.degree), key=lambda tup: tup[1], reverse=True)
            print("recompute", i, ntoremove)
        workingGraph.remove_node(deg.pop(0)[0])
    print("removed", nnodes - len(workingGraph), "goal", ntoremove)
    largest_cc = max(nx.connected_components(workingGraph), key=len)
    print(len(largest_cc))
    return len(largest_cc)

def init_random_walk(graph):
    sequence = set()
    randStart = random.randint(0, len(graph)-1)
    randStart = list(graph.nodes())[randStart]
    nnodes = math.ceil(len(graph)*0.1)
    sequence.add(randStart)
    while len(sequence) != nnodes:
        neighs = [n for n in graph.neighbors(randStart)]
        next = neighs[random.randint(0, len(neighs) - 1)]
        sequence.add(next)
        randStart = next
    return list(sequence)

def plotdeg(graph, name, colorchar):
    deg = [graph.degree(n) for n in graph.nodes()]
    degprob = dict(Counter(deg))
    for key in degprob:
        degprob[key] = degprob[key] / len(graph)
    if 0 in degprob.keys(): degprob.pop(0)
    print(sum(degprob.values()))

    plt.figure()
    plt.loglog(degprob.keys(), degprob.values(), colorchar + "o", label="degree")
    plt.legend()
    plt.xlabel("k")
    plt.ylabel("Pk")
    plt.title(name + " degree probability distribution")
    plt.show()


def compute_gamma(degrees, kmin):
    sum = 0.0
    ndot = 0
    for deg in degrees:
        if deg >= kmin:
            ndot += 1
            sum += math.log(deg/(kmin-0.5))
    return 1 + ndot*math.pow(sum, -1)

def test_plot_gamma(name, gamma, degrees, graph, kmin):
    degprob = dict(Counter(degrees))
    try:degprob.pop(0)
    except: pass
    for key in degprob.keys(): degprob[key] = math.pow(key, -gamma)
    #deggama = [math.pow(n, -gamma) for n in degprob]

    degprob2 = dict(Counter(degrees))
    for key in degprob2: degprob2[key] = degprob2[key] / len(graph)
    if 0 in degprob2.keys(): degprob2.pop(0)
    mintoplot = min(degprob2.values())
    topop = [key for key in degprob.keys() if degprob[key] < mintoplot]
    for key in topop: degprob.pop(key)

    plt.figure()
    plt.loglog(degprob.keys(), degprob.values(), "bo", label="power law y: " + str(round(gamma, 5)))
    plt.loglog(degprob2.keys(), degprob2.values(), "ro", label="degree")
    plt.xlabel("k")
    plt.ylabel("Pk")
    plt.legend()
    plt.title(name + " degree probability distribution and kmin: " + str(kmin))
    plt.show()