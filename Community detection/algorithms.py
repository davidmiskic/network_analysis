#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math

import networkx as nx
import random
import matplotlib.pyplot as plt
import cdlib
from cdlib import algorithms
from math import log, ceil
import random
from sklearn.metrics.cluster import normalized_mutual_info_score
from collections import Counter
from scipy.special import binom
import copy
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from helper import *

"""
Evaluate Leiden, Infomap and Walktrap community detection
"""
# variant of Girvan-Newman benchmark graphs with varying degrees of communities
def evaluate_gn():
    first = [i for i in range(0, 24)]
    second = [i for i in range(24, 48)]
    third = [i for i in range(48, 72)]
    origdist = [first, second, third]
    mis = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    nmis_scores = {"leiden": [], "infomap": [], "walktrap": []}
    for mi in mis:
        print(mi)
        leiden, infomap, walktrap = [], [], []
        for i in range(25):
            graph = build_GN_graph(72, 24, mi)
            leiden.append(NMIS(makedisjoint(runLeiden(graph)), origdist))
            infomap.append(NMIS(makedisjoint(runInfomap(graph)), origdist))
            walktrap.append(NMIS(makedisjoint(runWalktrap(graph)), origdist))
        nmis_scores["leiden"].append(sum(leiden) / len(leiden))
        nmis_scores["infomap"].append(sum(infomap) / len(infomap))
        nmis_scores["walktrap"].append(sum(walktrap) / len(walktrap))
    # nmis:Y, mi:X
    print(nmis_scores)
    plt.figure()
    plt.plot(mis, nmis_scores["leiden"], "r-", label="leiden")
    plt.plot(mis, nmis_scores["infomap"], "g-", label="infomap")
    plt.plot(mis, nmis_scores["walktrap"], "b-", label="walktrap")
    plt.legend()
    plt.xlabel("u")
    plt.ylabel("NMIS")
    plt.title("Girvan-Newman" + " normalized mutual information")
    plt.show()


# Erdos-Renyi random graph that lacks community structure
def evaluate_er():
    degs = [8, 16, 24, 32, 40]
    nvi_scores = {"leiden": [], "infomap": [], "walktrap": []}
    for deg in degs:

        leiden, infomap, walktrap = [], [], []
        for i in range(25):
            graph = build_ER_graph(deg)
            origdist = [list(c) for c in sorted(nx.connected_components(graph), key=len, reverse=False)]
            try:
                leiden.append(NVI(makedisjoint(runLeiden(graph)), origdist))
                infomap.append(NVI(makedisjoint(runInfomap(graph)), origdist))
                walktrap.append(NVI(makedisjoint(runWalktrap(graph)), origdist))
            except:
                print("EXP")
                return
        nvi_scores["leiden"].append(sum(leiden) / len(leiden))
        nvi_scores["infomap"].append(sum(infomap) / len(infomap))
        nvi_scores["walktrap"].append(sum(walktrap) / len(walktrap))

    print(nvi_scores)
    plt.figure()
    plt.plot(degs, nvi_scores["leiden"], "r-", label="leiden")
    plt.plot(degs, nvi_scores["infomap"], "g-", label="infomap")
    plt.plot(degs, nvi_scores["walktrap"], "b-", label="walktrap")
    plt.legend()
    plt.xlabel("deg")
    plt.ylabel("NVI")
    plt.title("ER" + " normalized variation of information")
    plt.show()



"""
Link prediction with preferential attachment, Adamic-Adar index, community index
"""
def link_prediction(graphh):
    import time
    AUCER = {"PAI":[], "AAI":[], "CI":[]}
    i = 0
    while i < 10:
        graph = copy.deepcopy(graphh)
        print(i)
        print(AUCER)
        try:
            start = time.time()
            AUC = framework(graph, 1)
            print("TIME ELAPSED PAI")
            print(time.time() - start)
            AUCER["PAI"].append(AUC)
        except: print("PAI ER EXP")
        graph = copy.deepcopy(graphh)
        try:
            start = time.time()
            AUC = framework(graph, 2)
            print("TIME ELAPSED AAI")
            print(time.time() - start)
            AUCER["AAI"].append(AUC)
        except: print("AAI ER EXP")
        graph = copy.deepcopy(graphh)
        try:
            start = time.time()
            AUC = framework(graph, 3)
            print("TIME ELAPSED CI")
            print(time.time() - start)
            AUCER["CI"].append(AUC)
        except: print("CI ER EXP")
        i+=1
    print(AUCER)
    for key in AUCER: AUCER[key] = sum(AUCER[key])/len(AUCER[key])
    print("FINAL=====================")
    print(AUCER)



"""
Detect communities in citation network
"""
def predict_citation_network():
    data = read_citation()
    train, test = make_test_train2013(data)
    Xtrain = train[:, [0, 2]]
    Ytrain = train[:, 1]
    Xtest = test[:, [0, 2]]
    Ytest = test[:, 1]

    clf = KNeighborsClassifier(n_neighbors=12000)
    clf.fit(Xtrain, Ytrain)
    Ypred = clf.predict(Xtest)
    CAKNN = accuracy_score(Ytest, Ypred)
    print("KNN", CAKNN)

    return CAKNN