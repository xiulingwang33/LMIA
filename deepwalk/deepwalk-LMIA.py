import networkx as nx
import numpy as np
import pickle as pk
import os
from gae.preprocessing import mask_test_edges
import deepwalk.deepwalk as DW
import pandas as pd
import sys
from bi_samplingv3 import *
import math

sys.setrecursionlimit(1000000)

seds=[1]
G_EGO_USERS=['dblp','fb','google+','cora','citeseer','pubmed','lastfm']
combs=[1,2,3,4,5,6,7]

for sed in seds:
    METHOD = 'bi-line%s' % (sed)

    for ego_user in G_EGO_USERS:

        feat_dir = './data/' + str(ego_user) + '-adj-feat.pkl'

        f2 = open(feat_dir, 'rb')

        adj, ft = pk.load(f2, encoding='latin1')

        g = nx.Graph(adj)


        if ego_user=='dblp' or  ego_user=='google+' :

            gindex=0
            for i, n in enumerate(g.nodes()):
                if (ft[n][gindex]==0):
                    ginfo = 1 #male
                elif (ft[n][gindex]==1):
                    ginfo = 2 #female

                else:
                    print('***')
                    ginfo = 0 #unknow gender

                g.nodes[n]['gender'] = ginfo


        elif ego_user=='fb':

            gindex = 77
            for i, n in enumerate(g.nodes()):
                if (ft[n][gindex] == 1 and ft[n][gindex + 1] != 1):
                    ginfo = 1  # male
                elif (ft[n][gindex + 1] == 1 and ft[n][gindex] != 1):
                    ginfo = 2  # female

                else:
                    print('***')
                    ginfo = 0  # unknow gender

                print(ginfo)

                g.nodes[n]['gender'] = ginfo

        else:

            with open('./data/' + str(ego_user) + '-target.txt') as tfile:
                Lines = tfile.readlines()
                target = []
                for line in Lines:
                    arr = line.strip().split(',')
                    target.append(int(arr[1]))

            for i, n in enumerate(g.nodes()):
                g.nodes[n]['gender'] = target[n]

        np.random.seed(sed)
        adj_sparse = nx.to_scipy_sparse_matrix(g)

        # Perform train-test split
        train_test_split = mask_test_edges(adj_sparse, test_frac=.3, val_frac=0)
        adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = train_test_split  # Unpack train-test split
        g_train = nx.from_scipy_sparse_matrix(
            adj_train)  # new graph object with only non-hidden edges, keep all the original nodes

        dp=0 ###dp=0:original, dp=1:differential, dp=5:adversarial

        if dp==1:
            sigma = 48
            F = 'dw-'+str(ego_user) +'-'+str(dp)+'-'+str(sigma)

        else:
            F = 'dw-'+str(ego_user) +'-'+ str(dp)


        res_dir = './data/'

        node_list=g.nodes()

        # for edge in train_edges:
        edge_tuples0 = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in g_train.edges()]

        train_edges0 = set(edge_tuples0)  # initialize train_edges to have all edges
        train_edges1 = np.array([list(edge_tuple) for edge_tuple in train_edges0])

        out = open('%s/%s-fair-%s-train.txt' % (res_dir, str(ego_user), F), 'w')
        for item in train_edges1:
            for jtem in item:
                out.write(str(jtem) + '\t')
            out.write('\n')
        out.close()


        train_edge_labels,test_edge_labels, emb_matrix,train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs,train_embs_1,train_embs_2,test_embs_1,test_embs_2 ,train_edges_sampled= DW.deepwalk8(g_train, train_test_split,
                                                                                           DATASET, METHOD, res_dir,
                                                                                           ego_user, F,dp,sigma)

        train_edges_list = np.array(train_edges_sampled)
        test_edges_list = test_edges

        edges_list = np.concatenate((train_edges_list, test_edges_list), axis=0)


        ylabel = [1] * train_sim_matrix.shape[0] + [0] * test_sim_matrix.shape[0]


        for comb in combs:
            if comb ==1:#dot
                sim_matrix = np.concatenate((train_sim_matrix[:, 0], test_sim_matrix[:, 0]), axis=0)

                sim_matrix_train = train_sim_matrix[:, 0]
                sim_matrix_test = test_sim_matrix[:, 0]

                sim_matrix = sim_matrix.reshape(-1, 1)
                # print(sim_matrix)

                sim_matrix_train = sim_matrix_train.reshape(-1, 1)
                sim_matrix_test = sim_matrix_test.reshape(-1, 1)

            if comb ==2:#cos
                sim_matrix = np.concatenate((train_sim_matrix[:, 1], test_sim_matrix[:, 1]), axis=0)
                sim_matrix_train = train_sim_matrix[:, 1]
                sim_matrix_test = test_sim_matrix[:, 1]

                sim_matrix = sim_matrix.reshape(-1, 1)

                sim_matrix_train = sim_matrix_train.reshape(-1, 1)
                sim_matrix_test = sim_matrix_test.reshape(-1, 1)

            if comb==3:#eu
                sim_matrix = np.concatenate((train_sim_matrix[:, 2], test_sim_matrix[:, 2]), axis=0)

                sim_matrix_train = train_sim_matrix[:, 2]
                sim_matrix_test = test_sim_matrix[:, 2]

                sim_matrix = sim_matrix.reshape(-1, 1)

                sim_matrix_train = sim_matrix_train.reshape(-1, 1)
                sim_matrix_test = sim_matrix_test.reshape(-1, 1)
            if comb == 4:#dot+cos
                sim_matrix = np.concatenate((train_sim_matrix[:, 0:2], test_sim_matrix[:, 0:2]), axis=0)

                sim_matrix_train = train_sim_matrix[:, 0:2]
                sim_matrix_test = test_sim_matrix[:, 0:2]
            if comb == 5:#dot+eu
                sim_matrix = np.concatenate((train_sim_matrix[:, [0,2]], test_sim_matrix[:, [0,2]]), axis=0)

                sim_matrix_train = train_sim_matrix[:, [0,2]]
                sim_matrix_test = test_sim_matrix[:, [0,2]]
            if comb == 6:#cos+eu
                sim_matrix = np.concatenate((train_sim_matrix[:, 1:3], test_sim_matrix[:, 1:3]), axis=0)

                sim_matrix_train = train_sim_matrix[:, 1:3]
                sim_matrix_test = test_sim_matrix[:, 1:3]

            if comb == 7:#dot+cos+eu
                sim_matrix = np.concatenate((train_sim_matrix[:, 0:3], test_sim_matrix[:, 0:3]), axis=0)

                sim_matrix_train = train_sim_matrix[:, 0:3]
                sim_matrix_test = test_sim_matrix[:, 0:3]


            from sklearn.cluster import KMeans
            from sklearn.metrics import accuracy_score

            accuracy = []
            for i in range(500):
                kmeans = KMeans(n_clusters=2, random_state=i).fit(sim_matrix)
                # kmeans = KMeans(n_clusters=2, random_state=i).fit(X)
                # print(kmeans.labels_)
                ylabel = [1] * train_sim_matrix.shape[0] + [0] * test_sim_matrix.shape[0]
                acc = accuracy_score(kmeans.labels_, ylabel)
                accuracy.append(acc)
            print(max(accuracy))

            acc_kmeans_sim = max(accuracy)

            tsts=[]
            print(len(kmeans.labels_))
            for i in range(len(kmeans.labels_)):
                node1=edges_list[i][0]
                node2=edges_list[i][1]
                dgr1=g.degree(node1)
                dgr2 = g.degree(node2)
                gender1 = g.nodes[node1]['gender']
                gender2 = g.nodes[node2]['gender']

                sim0 = sim_matrix[i]

                tst = [kmeans.labels_[i], ylabel[i], node1, node2, dgr1, dgr2, gender1, gender2]
                tsts.append(tst)
            name = ['y_score', 'y_test_grd', 'node1', 'node2', 'dgr1', 'dgr2', 'gender1', 'gender2']
            result = pd.DataFrame(columns=name, data=tsts)
            result.to_csv("{}{}-kmeans_sim_{}.csv".format(res_dir, F,comb))

            cents = kmeans.cluster_centers_

            dist0 = 0
            dist1 = 0

            for l in range(len(kmeans.labels_)):
                dist0 = np.sqrt(np.sum(np.square(sim_matrix[l] - cents[0])))
                dist1 = np.sqrt(np.sum(np.square(sim_matrix[l] - cents[1])))
                if kmeans.labels_[l] == 0 and (dist0 < dist1):
                    cent0 = cents[0]
                    cent1 = cents[1]
                    break
                elif kmeans.labels_[l] == 0 and (dist0 > dist1):
                    cent1 = cents[0]
                    cent0 = cents[1]
                    break

            dis0 = []
            dis1 = []

            for l in range(len(kmeans.labels_)):

                if kmeans.labels_[l] == 0:
                    dist = np.sqrt(np.sum(np.square(sim_matrix[l] - cent0)))
                    dis0.append(dist)

                else:
                    dist = np.sqrt(np.sum(np.square(sim_matrix[l] - cent1)))
                    dis1.append(dist)

            dist0 = sum(dis0) / len(dis0)
            dist1 = sum(dis1) / len(dis1)

            print(dist0, dist1)

            print(np.shape(ylabel))

            #
            from sklearn.model_selection import train_test_split

            ylabel1=ylabel
            ylable1=np.reshape(len(ylabel1),1)
            y_label=np.zeros((np.shape(edges_list)[0],3))
            for i in range(np.shape(edges_list)[0]):
                y_label[i][0]=edges_list[i][0]
                y_label[i][1] = edges_list[i][1]
                y_label[i][2] = ylabel[i]
            print(np.shape(y_label))

            y_label_train=np.zeros((np.shape(train_edges_list)[0],3))
            for i in range(np.shape(train_edges_list)[0]):
                y_label_train[i][0]=train_edges_list[i][0]
                y_label_train[i][1] = train_edges_list[i][1]
                y_label_train[i][2] = 1
            print(np.shape(y_label_train))

            y_label_test=np.zeros((np.shape(test_edges_list)[0],3))
            for i in range(np.shape(test_edges_list)[0]):
                y_label_test[i][0]=test_edges_list[i][0]
                y_label_test[i][1] = test_edges_list[i][1]
                y_label_test[i][2] = 0
            print(np.shape(y_label_test))



            X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(sim_matrix_train, y_label_train, test_size=0.3, random_state=42)

            X_test_train, X_test_test, y_test_train, y_test_test = train_test_split(sim_matrix_test, y_label_test,
                                                                                        test_size=0.3, random_state=42)

            X_train=np.concatenate((X_train_train, X_test_train),axis=0)
            X_test = np.concatenate((X_train_test, X_test_test), axis=0)
            y_train=np.concatenate((y_train_train, y_test_train),axis=0)
            y_test=np.concatenate((y_train_test, y_test_test),axis=0)


            #X_train, X_test, y_train, y_test = train_test_split(sim_matrix, y_label, test_size=0.3, random_state=42)
            #
            # # ######################################################################

            from sklearn import metrics
            from sklearn.neural_network import MLPClassifier

            mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16, 18), random_state=1,
                                max_iter=500)

            mlp.fit(X_train, y_train[:,2])

            print("Training set score: %f" % mlp.score(X_train, y_train[:,2]))
            print("Test set score: %f" % mlp.score(X_test, y_test[:,2]))

            y_score = mlp.predict(X_test)
            print(metrics.f1_score(y_test[:,2], y_score, average='micro'))
            print(metrics.classification_report(y_test[:,2], y_score, labels=range(3)))

            acc_mlp_sim = accuracy_score(y_score, y_test[:,2])

            tsts=[]
            for i in range(len(y_score)):
                node1=y_test[i][0]
                node2=y_test[i][1]
                dgr1=g.degree(node1)
                dgr2 = g.degree(node2)

                gender1 = g.nodes[node1]['gender']
                gender2 = g.nodes[node2]['gender']

                tst = [y_score[i], y_test[i][2], y_test[i][0], y_test[i][1], dgr1, dgr2, gender1, gender2]
                tsts.append(tst)
            name = ['y_score', 'y_test_grd', 'node1', 'node2', 'dgr1', 'dgr2', 'gender1', 'gender2']
            result = pd.DataFrame(columns=name, data=tsts)
            result.to_csv("{}{}-mlp_sim_{}.csv".format(res_dir, F,comb))

            # # ######################################################################

            from sklearn.ensemble import RandomForestClassifier

            rf = RandomForestClassifier(max_depth=150, random_state=0)
            rf.fit(X_train, y_train[:,2])

            print("Training set score: %f" % rf.score(X_train, y_train[:,2]))
            print("Test set score: %f" % rf.score(X_test, y_test[:,2]))

            y_score = rf.predict(X_test)
            print(metrics.f1_score(y_test[:,2], y_score, average='micro'))
            print(metrics.classification_report(y_test[:,2], y_score, labels=range(3)))

            acc_rf_sim = accuracy_score(y_score, y_test[:,2])


            tsts=[]
            for i in range(len(y_score)):
                node1=y_test[i][0]
                node2=y_test[i][1]
                dgr1=g.degree(node1)
                dgr2 = g.degree(node2)

                gender1 = g.nodes[node1]['gender']
                gender2 = g.nodes[node2]['gender']

                tst = [y_score[i], y_test[i][2], y_test[i][0], y_test[i][1], dgr1, dgr2, gender1, gender2]
                tsts.append(tst)
            name = ['y_score', 'y_test_grd', 'node1', 'node2', 'dgr1', 'dgr2', 'gender1', 'gender2']

            result = pd.DataFrame(columns=name, data=tsts)
            result.to_csv("{}{}-rf_sim_{}.csv".format(res_dir, F,comb))

            # # ######################################################################

            from sklearn.multiclass import OneVsRestClassifier
            from sklearn.svm import SVC

            svm = OneVsRestClassifier(SVC())
            svm.fit(X_train, y_train[:,2])

            print("Training set score: %f" % svm.score(X_train, y_train[:,2]))
            print("Test set score: %f" % svm.score(X_test, y_test[:,2]))

            y_score = svm.predict(X_test)
            print(metrics.f1_score(y_test[:,2], y_score, average='micro'))
            print(metrics.classification_report(y_test[:,2], y_score, labels=range(3)))

            acc_svm_sim = accuracy_score(y_score, y_test[:,2])


            tsts=[]
            for i in range(len(y_score)):
                node1=y_test[i][0]
                node2=y_test[i][1]
                dgr1=g.degree(node1)
                dgr2 = g.degree(node2)
                gender1 = g.nodes[node1]['gender']
                gender2 = g.nodes[node2]['gender']

                tst = [y_score[i], y_test[i][2], y_test[i][0], y_test[i][1], dgr1, dgr2, gender1, gender2]
                tsts.append(tst)
            name = ['y_score', 'y_test_grd', 'node1', 'node2', 'dgr1', 'dgr2', 'gender1', 'gender2']
            result = pd.DataFrame(columns=name, data=tsts)
            result.to_csv("{}{}-svm_sim_{}.csv".format(res_dir, F,comb))

