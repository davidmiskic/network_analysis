#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math

import networkx as nx
import cdlib
from cdlib import algorithms
from math import log, ceil
import os
import random
from scipy.special import binom
import copy
import numpy

def p_inout_GN(mi, n=72, m=24):
    # https: // math.stackexchange.com / questions / 1533360 / expected - degree - of - a - vertex - in -a - random - network
    p = (1 - mi) * (m - 1) + mi * (n - m)
    p = 20 / p
    pin = (1 - mi) * p
    pout = mi * p
    print(pin, pout, pin * 23 + pout * 48)
    return pin, pout


def build_GN_graph(n, m, mi):
    # n nodes, m size of module
    if n % m != 0: return None  # n = X*m
    pin, pout = p_inout_GN(mi, n, m)
    G = nx.Graph()
    G.add_nodes_from([i for i in range(0, n)])
    for i in range(0, n):
        for j in range(i + 1, n):
            # same module
            if i // m == j // m and random.random() < pin:
                G.add_edge(i, j)
            elif i // m != j // m and random.random() < pout:
                G.add_edge(i, j)
    # nx.draw(G, with_labels=True)
    # plt.show()
    return G


def sortedmodules(algout):
    # order modules by sum, so lowest is always with lowest index
    sums = []
    for item in algout: sums.append(sum(item))
    sums = sorted(sums)
    sorout = []
    for s in sums:
        for item in algout:
            if s == sum(item):
                sorout.append(item)
    # print(sorout)
    return sorout


def runLeiden(graph):
    part = algorithms.leiden(graph)
    return sortedmodules(part.communities)


def runInfomap(graph):
    im = algorithms.infomap(graph)
    return sortedmodules(im.communities)


def runWalktrap(graph):
    coms = algorithms.walktrap(graph)
    return sortedmodules(coms.communities)


def NMIS(algout, origdist):
    original = cdlib.NodeClustering(origdist, graph=None, method_name="")
    alg = cdlib.NodeClustering(algout, graph=None, method_name="")
    mod = original.adjusted_mutual_information(alg)
    # print(mod, mod.score)
    return mod.score

# taken from https://gist.github.com/jwcarr/626cbc80e0006b526688
def NVI(algout, origdist):
    def variation_of_information(X, Y):
        n = float(sum([len(x) for x in X]))
        sigma = 0.0
        for x in X:
            p = len(x) / n
            for y in Y:
                q = len(y) / n
                r = len(set(x) & set(y)) / n
                if r > 0.0:
                    sigma += r * (log(r / p, 2) + log(r / q, 2))
        return abs(sigma)

    nelem = sum([len(x) for x in algout])
    return variation_of_information(origdist, algout) / math.log(nelem)


def makedisjoint(algout):
    newout = []
    used = set()
    for community in algout:
        newComm = []
        for item in community:
            if item not in used:
                newComm.append(item)
                used.add(item)
        if len(newComm) > 0: newout.append(newComm)
    return newout


def build_ER_graph(avgDeg, n=1000):
    # 2m/n = avgDeg, 2m= avgDeg*n
    m = round((avgDeg * n) / 2)
    G = nx.gnm_random_graph(n, m)
    return G


def framework(graph, s):
    def PAI(n1, n2):
        return graph.degree[n1] * graph.degree[n2]

    def AAI(n1, n2):
        data = nx.adamic_adar_index(graph, ebunch=[(n1, n2)])
        for u, v, p in data: return p

    def CI(comms, n1, n2):
        comn1 = [x for x in comms if n1 in x][0]
        comn2 = [x for x in comms if n2 in x][0]
        if comn1 != comn2: return 0
        mc = graph.subgraph(comn1).number_of_edges()
        nc = len(comn1)
        return mc / binom(nc, 2)

    def AUC(comms = None):
        Lpp = [random.choice(Lp) for i in range(len(Lp))]
        Lnn = [random.choice(Ln) for i in range(len(Ln))]
        if s == 1:
            Lpp = [PAI(*e) for e in Lpp]
            Lnn = [PAI(*e) for e in Lnn]
        elif s == 2:
            Lpp = [AAI(*e) for e in Lpp]
            Lnn = [AAI(*e) for e in Lnn]
        elif s == 3:
            Lpp = [CI(comms, *e) for e in Lpp]
            Lnn = [CI(comms, *e) for e in Lnn]

        m1 = 0
        m2 = 0
        for x in range(round(m/10)):
            item1 = random.choice(Lpp)
            item2 = random.choice(Lnn)
            if item1 > item2:
                m1 += 1
            elif item1 == item2:
                m2 += 1
        AUC = (m1 + (m2 / 2)) / round(m / 10)
        return AUC

    Ln = []
    m = graph.number_of_edges()
    nodes = list(graph.nodes)
    while len(Ln) != round(m/10):
        node1 = random.choice(nodes)
        node2 = random.choice(nodes)
        if not graph.has_edge(node1, node2) and node1 != node2 and (node1, node2) not in Ln: Ln.append((node1, node2))
    print("Ln")
    choices = [e for e in graph.edges]
    Lp = []
    while len(Lp) != round(m / 10):
        c = random.choice(choices)
        if c not in Lp and graph.has_edge(*c):
            graph.remove_edge(*c)
            Lp.append(c)
    print("Lp")
    if s == 3:
        return AUC(comms = makedisjoint(runLeiden(graph)))
    else: return AUC()

def make_test_train2013(data):
    testind = []
    for i in range(0, numpy.shape(data)[0]):
        if data[i, 2] == 2013: testind.append(i)
    test_data = None
    for i in testind:
        if hasattr(test_data, 'shape'):
            test_data = numpy.vstack([test_data, data[i, :]])
        else:
            test_data = numpy.array([data[i, :]])

    train_data = None
    for i in range(0, numpy.shape(data)[0]):
        if i not in testind:
            if hasattr(train_data, 'shape'):
                train_data = numpy.vstack([train_data, data[i, :]])
            else:
                train_data = numpy.array([data[i, :]])

    return train_data, test_data

def read_citation():
    def readDist(file="aps_2008_2013"):
        f = open(file, "r")
        for i in range(4): f.readline()
        line = f.readline()
        A = None
        while line:
            item = line.replace("\n", "").replace('"', "").split(" ")
            if item == ["#"]: break
            node, key, year = int(item[1]), int(item[3]), int(item[2].split("-")[1])
            if hasattr(A, 'shape'): A = numpy.vstack([A, [node, key, year]])
            else: A = numpy.array([[node, key, year]])
            line = f.readline()
        f.close()
        return A
    real = readDist()
    return real#, nx.from_edgelist("aps_2008_2013", nodetype=int)