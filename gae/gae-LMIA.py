import networkx as nx
import numpy as np
import pickle as pk
# import os
# from gae.preprocessing import mask_test_edges
# import deepwalk.deepwalk as DW
# # import link_prediction_scores
import pandas as pd
import sys
# import gurobipy as gp
# from gurobipy import GRB
# from new_sampling_methods import *
# from edge_sampling import *
# from base_sampling_methods import *
# from dynamic_sampling import *
# from bi_samplingv3 import *
import math
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import random
import itertools
import matplotlib.pyplot as plt
import scipy.optimize as optimize

sys.setrecursionlimit(1000000)



seds=[1]
#METHOD = 'bi-deepwalk2'
DATASET = 'facebook-data-new-2'
def get_edge_embeddings(edge_list, emb_matrixs):
    embs = []
    i = 0
    for edge in edge_list:
        node1 = int(edge[0])
        node2 = int(edge[1])
        emb = []
        # print(i)
        # print(idx_epoches_all[i,:])
        # print(len(idx_epoches_all[i,:]))

        emb1 = emb_matrixs[node1]
        # print(np.shape(emb1))
        emb2 = emb_matrixs[node2]
        edge_emb = np.multiply(emb1, emb2)
        sim1 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 0.0000000000000000000000000000001)

        sim2 = np.dot(emb1, emb2)

        sim3 = np.linalg.norm(np.array(emb1) - np.array(emb2))

        # edge_emb = np.array(emb1) + np.array(emb2)
        # print(np.shape(edge_emb))
        # emb.append(sim1)
        # emb.append(sim2)
        i += 1
        embs.append([sim1, sim2, sim3])
    embs = np.array(embs)
    return embs



def target_func(x,a0,a1,a2):
    return a0*np.exp(-x/a1)+a2

def target_func2(x,a,b,c):
    return b*np.power(a,x)+c

def target_func3(x,a,b):
    return a*x+b

def target_func4(x,a,b,c):
    return a*x*x+b*x+c

# G_EGO_USERS=['3980','698','0','107','348','1684','1912','3437','414','686']
G_EGO_USERS=['Facebook','cora','citeseer','dblp','lastfm','pubmed']
G_EGO_USERS=['citeseer','Facebook','cora','lastfm']
G_EGO_USERS=['pubmed','dblp']


result_all = []

results_avg=[]
for sed in seds:
    for ego_user in G_EGO_USERS:
        # feature

        feat_dir = './data/' + str(ego_user) + '-adj-feat.pkl'

        f2 = open(feat_dir, 'rb')

        adj, ft = pk.load(f2, encoding='latin1')

        g = nx.Graph(adj)


        res_dir = '%s/'%(ego_user)

        METHOD = 'gae'
        dp = 0
        sigma = 0
        if dp == 1:
            F = str(ego_user) + '-' + str(dp) + '-' + str(sigma)

        else:
            F = str(ego_user) + '-' + str(dp)

        f3 = res_dir + METHOD + '-embeds-' + F + '-' + str(ego_user) + '.npy'
        emb_matrix = np.load(f3)

        dt=ego_user

        res_dir0 = './%s/' % (dt)

        f2 = open('%s/%s-train_test_split' % (res_dir0, dt), 'rb')
        train_test_split = pk.load(f2, encoding='latin1')

        adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = train_test_split  # Unpack train-test split
        adj = adj_train

        edgeall = list([list(edge_tuple) for edge_tuple in train_edges])

        train_edges_list = random.sample(edgeall, np.shape(test_edges)[0])
        test_edges_list=test_edges

        if np.shape(train_edges_list)[0] > 100000:
            num = np.shape(train_edges_list)[0]
            sampel_idx = np.array(random.sample(range(num), 100000))
            train_edges_list = train_edges_list[sampel_idx]
            test_edges_list = test_edges_list[sampel_idx]

        edges_list = np.concatenate((train_edges_list, test_edges_list), axis=0)

        train_sim_matrix = get_edge_embeddings(train_edges_list, emb_matrix)
        test_sim_matrix = get_edge_embeddings(test_edges_list, emb_matrix)

        savepath = res_dir + 'mem-sim-www' + str(ego_user)
        print('***')
        np.save(savepath, train_sim_matrix)

        savepath = res_dir + 'non-mem-sim-www' + str(ego_user)
        print('***')
        np.save(savepath, test_sim_matrix)


        ylabel = [1] * train_sim_matrix.shape[0] + [0] * test_sim_matrix.shape[0]

        # print(train_sim_matrix)
        # print(test_sim_matrix)

        sim_matrix = np.concatenate((train_sim_matrix, test_sim_matrix), axis=0)
        # print(sim_matrix)
        print(np.shape(train_sim_matrix))
        print(np.shape(test_sim_matrix))
        # sim_matrix = sim_matrix
        # print(sim_matrix)
        print(np.shape(sim_matrix))
        # exit()

        sim_matrix_train = train_sim_matrix
        sim_matrix_test = test_sim_matrix

        from sklearn.cluster import KMeans
        from sklearn.metrics import accuracy_score

        accuracy = []
        for i in range(500):
            kmeans = KMeans(n_clusters=2, random_state=i).fit(sim_matrix)

            ylabel = [1] * train_sim_matrix.shape[0] + [0] * test_sim_matrix.shape[0]
            acc = accuracy_score(kmeans.labels_, ylabel)
            accuracy.append(acc)
        print(max(accuracy))

        acc_kmeans_sim = max(accuracy)

        tsts = []
        print(len(kmeans.labels_))
        for i in range(len(kmeans.labels_)):
            node1 = edges_list[i][0]
            node2 = edges_list[i][1]
            # dgr1 = g.degree(node1)
            # dgr2 = g.degree(node2)
            # gender1 = g.nodes[node1]['gender']
            # gender2 = g.nodes[node2]['gender']

            sim0 = sim_matrix[i]

            tst = [kmeans.labels_[i], ylabel[i], node1, node2]
            tsts.append(tst)
        name = ['y_score', 'y_test_grd', 'node1', 'node2']
        result = pd.DataFrame(columns=name, data=tsts)
        result.to_csv("{}{}-kmeans_sim.csv".format(res_dir, F))


        from sklearn.model_selection import train_test_split

        ylabel1 = ylabel
        ylable1 = np.reshape(len(ylabel1), 1)

        y_label = np.zeros((np.shape(edges_list)[0], 3))
        for i in range(np.shape(edges_list)[0]):
            y_label[i][0] = edges_list[i][0]
            y_label[i][1] = edges_list[i][1]
            y_label[i][2] = ylabel[i]
        print(np.shape(y_label))

        y_label_train = np.zeros((np.shape(train_edges_list)[0], 3))
        for i in range(np.shape(train_edges_list)[0]):
            y_label_train[i][0] = train_edges_list[i][0]
            y_label_train[i][1] = train_edges_list[i][1]
            y_label_train[i][2] = 1
        print(np.shape(y_label_train))

        y_label_test = np.zeros((np.shape(test_edges_list)[0], 3))
        for i in range(np.shape(test_edges_list)[0]):
            y_label_test[i][0] = test_edges_list[i][0]
            y_label_test[i][1] = test_edges_list[i][1]
            y_label_test[i][2] = 0
        print(np.shape(y_label_test))

        X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(sim_matrix_train, y_label_train,
                                                                                    test_size=0.3, random_state=42)

        X_test_train, X_test_test, y_test_train, y_test_test = train_test_split(sim_matrix_test, y_label_test,
                                                                                test_size=0.3, random_state=42)

        X_train = np.concatenate((X_train_train, X_test_train), axis=0)
        X_test = np.concatenate((X_train_test, X_test_test), axis=0)
        y_train = np.concatenate((y_train_train, y_test_train), axis=0)
        y_test = np.concatenate((y_train_test, y_test_test), axis=0)

        # X_train, X_test, y_train, y_test = train_test_split(sim_matrix, y_label, test_size=0.3, random_state=42)
        #
        # # ######################################################################

        from sklearn import metrics
        from sklearn.neural_network import MLPClassifier

        mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16, 18), random_state=1,
                            max_iter=500)

        mlp.fit(X_train, y_train[:, 2])

        print("Training set score: %f" % mlp.score(X_train, y_train[:, 2]))
        print("Test set score: %f" % mlp.score(X_test, y_test[:, 2]))

        y_score = mlp.predict(X_test)
        print(metrics.f1_score(y_test[:, 2], y_score, average='micro'))
        print(metrics.classification_report(y_test[:, 2], y_score, labels=range(3)))

        proba = mlp.predict_proba(X_test)
        # proba = np.amax(proba, axis=1)
        proba = proba[:, 1]

        acc_mlp_sim = accuracy_score(y_score, y_test[:, 2])

        y_label_test=y_test

        acc = accuracy_score(y_label_test[:, 2], y_score)
        recall = recall_score(y_score, y_label_test[:, 2])
        precision = precision_score(y_score, y_label_test[:, 2])
        f1 = f1_score(y_score, y_label_test[:, 2])
        auc = roc_auc_score(y_label_test[:, 2], proba)

        print(ego_user, acc, recall, precision, f1, auc)

        tsts = []
        for i in range(len(y_score)):
            node1 = y_test[i][0]
            node2 = y_test[i][1]
            # dgr1 = g.degree(node1)
            # dgr2 = g.degree(node2)
            #
            # gender1 = g.nodes[node1]['gender']
            # gender2 = g.nodes[node2]['gender']

            tst = [y_score[i], y_test[i][2], y_test[i][0], y_test[i][1]]
            tsts.append(tst)
        name = ['y_score', 'y_test_grd', 'node1', 'node2']
        result = pd.DataFrame(columns=name, data=tsts)
        result.to_csv("{}{}-mlp_sim.csv".format(res_dir, F))

        # # ######################################################################

        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(max_depth=150, random_state=0)
        rf.fit(X_train, y_train[:, 2])

        print("Training set score: %f" % rf.score(X_train, y_train[:, 2]))
        print("Test set score: %f" % rf.score(X_test, y_test[:, 2]))

        y_score = rf.predict(X_test)
        print(metrics.f1_score(y_test[:, 2], y_score, average='micro'))
        print(metrics.classification_report(y_test[:, 2], y_score, labels=range(3)))

        y_label_test=y_test

        proba = rf.predict_proba(X_test)
        # proba = np.amax(proba, axis=1)
        proba = proba[:, 1]

        acc_rf_sim = accuracy_score(y_score, y_test[:, 2])

        acc = accuracy_score(y_label_test[:, 2], y_score)
        recall = recall_score(y_score, y_label_test[:, 2])
        precision = precision_score(y_score, y_label_test[:, 2])
        f1 = f1_score(y_score, y_label_test[:, 2])
        auc = roc_auc_score(y_label_test[:, 2], proba)

        print(ego_user, acc, recall, precision, f1, auc)

        tsts = []
        for i in range(len(y_score)):
            node1 = y_test[i][0]
            node2 = y_test[i][1]
            # dgr1 = g.degree(node1)
            # dgr2 = g.degree(node2)
            #
            # gender1 = g.nodes[node1]['gender']
            # gender2 = g.nodes[node2]['gender']

            tst = [y_score[i], y_test[i][2], y_test[i][0], y_test[i][1]]
            tsts.append(tst)
        name = ['y_score', 'y_test_grd', 'node1', 'node2']

        result = pd.DataFrame(columns=name, data=tsts)
        result.to_csv("{}{}-rf_sim.csv".format(res_dir, F))
