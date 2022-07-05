#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
from io import open
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import logging

from . import graph
from . import walks as serialized_walks
from gensim.models import Word2Vec
from .skipgram import Skipgram

from six import text_type as unicode
from six import iteritems
from six.moves import range
import tensorflow as tf
import numpy as np
import argparse

import pickle
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

import psutil
import networkx as nx

from multiprocessing import cpu_count

p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def debug(type_, value, tb):
  if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    sys.__excepthook__(type_, value, tb)
  else:
    import traceback
    import pdb
    traceback.print_exception(type_, value, tb)
    print(u"\n")
    pdb.pm()


def write_walks(args, walks,DATASET,METHOD,F,ego_user):
    file_ = open('E:/python/banlance/code/' + DATASET + '/' + METHOD + '/walks/' + F + '-' + str(ego_user), 'w')
    for walk in walks:
        line = str()
        for node in walk:
            line += str(node) + ' '
        line += '\n'
        file_.write(line)
    file_.close()


def process(args,adj_train,DATASET,METHOD,F,ego_user):

  if args.format == "adjlist":
    G = graph.load_adjacencylist(args.input, undirected=args.undirected)
  elif args.format == "edgelist":
    G = graph.load_edgelist(args.input, undirected=args.undirected)
  elif args.format == "mat":
    G = graph.load_matfile(args.input, variable_name=args.matfile_variable_name, undirected=args.undirected)
  else:
    raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', 'mat'" % args.format)

  print("Number of nodes: {}".format(len(G.nodes())))

  num_walks = len(G.nodes()) * args.number_walks

  print("Number of walks: {}".format(num_walks))

  data_size = num_walks * args.walk_length

  print("Data size (walks*length): {}".format(data_size))

  if data_size < args.max_memory_data_size:
    print("Walking...")
    walks = graph.build_deepwalk_corpus(G, num_paths=args.number_walks,
                                        path_length=args.walk_length, alpha=0, rand=random.Random(args.seed))

    write_walks(args, walks, DATASET, METHOD, F, ego_user)
    print("Training...")
    model = Word2Vec(walks, size=args.representation_size, window=args.window_size, min_count=0, sg=1, hs=1, workers=args.workers)
  else:
    print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size, args.max_memory_data_size))
    print("Walking...")

    walks_filebase = args.output + ".walks"
    walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=args.number_walks,
                                         path_length=args.walk_length, alpha=0, rand=random.Random(args.seed),
                                         num_workers=args.workers)

    print("Counting vertex frequency...")
    if not args.vertex_freq_degree:
      vertex_counts = serialized_walks.count_textfiles(walk_files, args.workers)
    else:
      # use degree distribution for frequency in tree
      vertex_counts = G.degree(nodes=G.iterkeys())

    print("Training...")
    walks_corpus = serialized_walks.WalksCorpus(walk_files)
    model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
                     size=args.representation_size,
                     window=args.window_size, min_count=0, trim_rule=None, workers=args.workers)

  model.wv.save_word2vec_format(args.output)



  # Store embeddings mapping
  emb_mappings = model.wv

  # Create node embeddings matrix (rows = nodes, columns = embedding features)
  emb_list = []
  for node_index in range(0, adj_train.shape[0]):
      node_str = str(node_index)
      node_emb = emb_mappings[node_str]
      emb_list.append(node_emb)
  emb_matrix = np.vstack(emb_list)

  with open('E:/python/banlance/code/' + DATASET + '/' + METHOD + '/embeds/' + F + '-' + str(ego_user), 'w') as f:
      f.write('%d %d\n' % (adj_train.shape[0], args.representation_size))
      for i in range(adj_train.shape[0]):
          e = ' '.join(map(lambda x: str(x), emb_list[i]))
          f.write('%s %s\n' % (str(i), e))

  return emb_matrix



def deepwalk(g_train, train_test_split,DATASET,METHOD,res_dir, ego_user, F):
  parser = ArgumentParser("deepwalk",
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          conflict_handler='resolve')

  parser.add_argument("--debug", dest="debug", action='store_true', default=False,
                      help="drop a debugger if an exception is raised.")

  parser.add_argument('--format', default='edgelist',
                      help='File format of input file')

  parser.add_argument('--input', nargs='?',
                      help='Input graph file')

  parser.add_argument("-l", "--log", dest="log", default="INFO",
                      help="log verbosity level")

  parser.add_argument('--matfile-variable-name', default='network',
                      help='variable name of adjacency matrix inside a .mat file.')

  parser.add_argument('--max-memory-data-size', default=1000000000, type=int,
                      help='Size to start dumping walks to disk, instead of keeping them in memory.')

  parser.add_argument('--number-walks', default=10, type=int,
                      help='Number of random walks to start at each node')

  parser.add_argument('--output',
                      help='Output representation file')

  parser.add_argument('--representation-size', default=128, type=int,
                      help='Number of latent dimensions to learn for each node.')

  parser.add_argument('--seed', default=0, type=int,
                      help='Seed for random walk generator.')

  parser.add_argument('--undirected', default=True, type=bool,
                      help='Treat graph as undirected.')

  parser.add_argument('--vertex-freq-degree', default=False, action='store_true',
                      help='Use vertex degree to estimate the frequency of nodes '
                           'in the random walks. This option is faster than '
                           'calculating the vocabulary.')

  parser.add_argument('--walk-length', default=80, type=int,
                      help='Length of the random walk started at each node')

  parser.add_argument('--window-size', default=10, type=int,
                      help='Window size of skipgram model.')

  parser.add_argument('--workers', default=10, type=int,
                      help='Number of parallel processes.')
  parser.add_argument('--edge_score_mode', default='edge-emb')


  args = parser.parse_args()
  args.input='%s/edgelist/%s-fair-%s-train.txt' % (res_dir, str(ego_user), F)
  args.output = 'E:/python/banlance/code/'+DATASET+'/'+METHOD+'/embeds/'+F+'-'+str(ego_user)
  numeric_level = getattr(logging, args.log.upper(), None)
  logging.basicConfig(format=LOGFORMAT)
  logger.setLevel(numeric_level)

  if args.debug:
   sys.excepthook = debug

  adj_train_orig, train_edges, train_edges_false, val_edges, val_edges_false, \
  test_edges, test_edges_false = train_test_split
  g_train=nx.adjacency_matrix(g_train)
  emb_matrix=process(args,g_train,DATASET,METHOD,F,ego_user)

  n2v_scores, val_edge_labels, val_preds, test_edge_labels, test_preds=linkpre_scores(args, emb_matrix, train_test_split, ego_user,DATASET,METHOD, F)

  return n2v_scores, val_edge_labels, val_preds, test_edge_labels, test_preds




def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def linkpre_scores(args, emb_matrix, train_test_split, ego_user,DATASET,METHOD, Flag):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = train_test_split
    start_time = time.time()
    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
    if args.edge_score_mode == "edge-emb":

        def get_edge_embeddings(edge_list,ego_user,DATASET, Flag, flag):
            embs = []
            for edge in edge_list:
                node1 = edge[0]
                node2 = edge[1]
                emb1 = emb_matrix[node1]
                #print(np.shape(emb1))
                emb2 = emb_matrix[node2]
                edge_emb = np.multiply(emb1, emb2)
                #edge_emb = np.array(emb1) + np.array(emb2)
                print(np.shape(edge_emb))
                embs.append(edge_emb)
            embs = np.array(embs)

            #with open('/Users/xiulingwang/Downloads/line-master/data/embds/' + str(ego_user) + flag + '-' + Flag, 'w') as f:
            with open('E:/python/banlance/code/' + DATASET + '/' + METHOD + '/embeds/' + Flag + '-' + str(ego_user) + flag,'w') as f:
            #with open('/Users/xiulingwang/Downloads/'+DATASET+'/line/3-split/' + str(ego_user)+ '-' + flag + '-' + Flag+'-' +'embds','w') as f:
                f.write('%d %d\n' % (edge_list.shape[0], args.representation_size))
                for i in range(edge_list.shape[0]):
                    e = ' '.join(map(lambda x: str(x), embs[i]))
                    f.write('%s %s %s\n' % (str(edge_list[i][0]), str(edge_list[i][1]), e))

            return embs

        # Train-set edge embeddings
        pos_train_edge_embs = get_edge_embeddings(train_edges,ego_user,DATASET, Flag, flag='pos-train')
        neg_train_edge_embs = get_edge_embeddings(train_edges_false, ego_user,DATASET,Flag, flag='neg-train')
        train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

        # Create train-set edge labels: 1 = real edge, 0 = false edge
        train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

        # Val-set edge embeddings, labels
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            pos_val_edge_embs = get_edge_embeddings(val_edges,ego_user,DATASET,Flag, flag='pos-val')
            neg_val_edge_embs = get_edge_embeddings(val_edges_false,ego_user,DATASET,Flag, flag='neg-val')
            val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
            val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

        # Test-set edge embeddings, labels
        pos_test_edge_embs = get_edge_embeddings(test_edges,ego_user,DATASET,Flag, flag='pos-test')
        neg_test_edge_embs = get_edge_embeddings(test_edges_false,ego_user,DATASET,Flag, flag='neg-test')
        test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

        # Create val-set edge labels: 1 = real edge, 0 = false edge
        test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

        # Train logistic regression classifier on train-set edge embeddings
        edge_classifier = LogisticRegression(random_state=0)
        edge_classifier.fit(train_edge_embs, train_edge_labels)

        # Predicted edge scores: probability of being of class "1" (real edge)
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
        test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
        print(test_preds)
        print(np.shape(test_preds))

        runtime = time.time() - start_time

        # Calculate scores
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            n2v_val_roc = roc_auc_score(val_edge_labels, val_preds)
            # n2v_val_roc_curve = roc_curve(val_edge_labels, val_preds)
            n2v_val_ap = average_precision_score(val_edge_labels, val_preds)
        else:
            n2v_val_roc = None
            n2v_val_roc_curve = None
            n2v_val_ap = None

        n2v_test_roc = roc_auc_score(test_edge_labels, test_preds)
        # n2v_test_roc_curve = roc_curve(test_edge_labels, test_preds)
        n2v_test_ap = average_precision_score(test_edge_labels, test_preds)


    # Generate edge scores using simple dot product of node embeddings (like in GAE paper)
    elif args.edge_score_mode == "dot-product":
        score_matrix = np.dot(emb_matrix, emb_matrix.T)
        runtime = time.time() - start_time

        # Val set scores
        if len(val_edges) > 0:
            n2v_val_roc, n2v_val_ap = get_roc_score(val_edges, val_edges_false, score_matrix, apply_sigmoid=True)
        else:
            n2v_val_roc = None
            n2v_val_roc_curve = None
            n2v_val_ap = None

        # Test set scores
        n2v_test_roc, n2v_test_ap = get_roc_score(test_edges, test_edges_false, score_matrix, apply_sigmoid=True)

    else:
        print
        "Invalid edge_score_mode! Either use edge-emb or dot-product."

    # Record scores
    n2v_scores = {}

    n2v_scores['test_roc'] = n2v_test_roc
    # n2v_scores['test_roc_curve'] = n2v_test_roc_curve
    n2v_scores['test_ap'] = n2v_test_ap

    n2v_scores['val_roc'] = n2v_val_roc
    # n2v_scores['val_roc_curve'] = n2v_val_roc_curve
    n2v_scores['val_ap'] = n2v_val_ap

    n2v_scores['runtime'] = runtime

    return n2v_scores, val_edge_labels, val_preds, test_edge_labels, test_preds


# Input: positive test/val edges, negative test/val edges, edge score matrix
# Output: ROC AUC score, ROC Curve (FPR, TPR, Thresholds), AP score
def get_roc_score(edges_pos, edges_neg, score_matrix, apply_sigmoid=False):
    # Edge case
    if len(edges_pos) == 0 or len(edges_neg) == 0:
        return (None, None, None)

    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        if apply_sigmoid == True:
            preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]]))
        else:
            preds_pos.append(score_matrix[edge[0], edge[1]])
        pos.append(1)  # actual value (1 for positive)

    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        if apply_sigmoid == True:
            preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]]))
        else:
            preds_neg.append(score_matrix[edge[0], edge[1]])
        neg.append(0)  # actual value (0 for negative)

    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    # roc_curve_tuple = roc_curve(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    # return roc_score, roc_curve_tuple, ap_score
    return roc_score, ap_score


# Return a list of tuples (node1, node2) for networkx link prediction evaluation
def get_ebunch(train_test_split):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = train_test_split

    test_edges_list = test_edges.tolist()  # convert to nested list
    test_edges_list = [tuple(node_pair) for node_pair in test_edges_list]  # convert node-pairs to tuples
    test_edges_false_list = test_edges_false.tolist()
    test_edges_false_list = [tuple(node_pair) for node_pair in test_edges_false_list]
    return (test_edges_list + test_edges_false_list)


