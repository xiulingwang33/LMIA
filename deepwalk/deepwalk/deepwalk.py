#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
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
#import tensorflow as tf
import numpy as np
import argparse

import pickle
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

import psutil
import networkx as nx
import torch
import torch.nn as nn

from word2vec import data_reader, model, trainer

from deepwalk_pytorch import word2vec

from multiprocessing import cpu_count

import pandas as pd

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
    if ego_user=='3980' or ego_user=='698':
        file_ = open('./' + METHOD + '-walks-' + F + '-' + str(ego_user), 'w')

    else:
        file_ = open('E:/python/banlance/code/' + DATASET + '/' + METHOD + '-walks-' + F + '-' + str(ego_user), 'w')
    for walk in walks:
        line = str()
        for node in walk:
            line += str(node) + ' '
        line += '\n'
        file_.write(line)
    file_.close()

def write_walks2(args, walks,DATASET,METHOD,F,ego_user,res_dir):
    #file_ = open('E:/python/banlance/code/' + DATASET + '/' + METHOD + '-walks-' + F + '-' + str(ego_user), 'w')
    if ego_user=='3980' or ego_user=='698':
        file_ = open('./' + METHOD + '-walks-' + F + '-' + str(ego_user), 'w')
    else:
        file_ = open(res_dir+ METHOD + '-walks-' + F + '-' + str(ego_user), 'w')
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

    training_loss = model.get_latest_training_loss()

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

  # exit()



  # Store embeddings mapping
  emb_mappings = model.wv

  # Create node embeddings matrix (rows = nodes, columns = embedding features)
  emb_list = []
  for node_index in range(0, adj_train.shape[0]):
      node_str = str(node_index)
      node_emb = emb_mappings[node_str]
      emb_list.append(node_emb)
  emb_matrix = np.vstack(emb_list)

  with open('E:\\python\\banlance\\code\\' + DATASET + '\\' + METHOD + '-embeds-' + F + '-' + str(ego_user), 'w') as f:
      f.write('%d %d\n' % (adj_train.shape[0], args.representation_size))
      for i in range(adj_train.shape[0]):
          e = ' '.join(map(lambda x: str(x), emb_list[i]))
          f.write('%s %s\n' % (str(i), e))

  return emb_matrix



def process_inf_debias(args,adj_train,DATASET,METHOD,F,ego_user):

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

    training_loss = model.get_latest_training_loss()

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

  # exit()



  # Store embeddings mapping
  emb_mappings = model.wv

  # Create node embeddings matrix (rows = nodes, columns = embedding features)
  emb_list = []
  for node_index in range(0, adj_train.shape[0]):
      node_str = str(node_index)
      node_emb = emb_mappings[node_str]
      emb_list.append(node_emb)
  emb_matrix = np.vstack(emb_list)

  with open('E:\\python\\banlance\\code\\' + DATASET + '\\' + METHOD + '-embeds-' + F + '-' + str(ego_user), 'w') as f:
      f.write('%d %d\n' % (adj_train.shape[0], args.representation_size))
      for i in range(adj_train.shape[0]):
          e = ' '.join(map(lambda x: str(x), emb_list[i]))
          f.write('%s %s\n' % (str(i), e))

  return emb_matrix







def deepwalk_inf_debias(g_train, train_test_split,DATASET, METHOD,res_dir, ego_user, F):
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
  args.input='%s/%s-fair-%s-train.txt' % (res_dir, str(ego_user), F)
  args.output = 'E:\\python\\banlance\\code\\'+DATASET+'\\'+METHOD+'-embeds-'+F+'-'+str(ego_user)
  numeric_level = getattr(logging, args.log.upper(), None)
  logging.basicConfig(format=LOGFORMAT)
  logger.setLevel(numeric_level)

  if args.debug:
   sys.excepthook = debug

  adj_train, train_edges, train_edges_false, test_edges, test_edges_false = train_test_split
  g_train=nx.adjacency_matrix(g_train)
  emb_matrix=process_inf_debias(args,g_train,DATASET,METHOD,F,ego_user)

  test_edge_labels, test_preds=linkpre_scores_inf_debias(args, emb_matrix, train_test_split, ego_user,DATASET,METHOD, F)

  return test_edge_labels,test_preds





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
  args.input='%s/%s-fair-%s-train.txt' % (res_dir, str(ego_user), F)
  args.output = 'E:\\python\\banlance\\code\\'+DATASET+'\\'+METHOD+'-embeds-'+F+'-'+str(ego_user)
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



def deepwalk1(g_train, train_test_split,DATASET,METHOD,res_dir, ego_user, F):
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
  args.input='%s%s-fair-%s-train.txt' % (res_dir, str(ego_user), F)
  args.output = res_dir+METHOD+'-embeds-'+F+'-'+str(ego_user)
  numeric_level = getattr(logging, args.log.upper(), None)
  logging.basicConfig(format=LOGFORMAT)
  logger.setLevel(numeric_level)

  if args.debug:
   sys.excepthook = debug

  adj_train_orig, train_edges, train_edges_false, val_edges, val_edges_false, \
  test_edges, test_edges_false = train_test_split
  g_train=nx.adjacency_matrix(g_train)
  emb_matrix=process(args,g_train,DATASET,METHOD,F,ego_user)

  n2v_scores,train_edge_labels, test_edge_labels, test_preds,train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs,train_embs_1,train_embs_2,test_embs_1,test_embs_2=linkpre_scores1(args, emb_matrix, train_test_split, ego_user,DATASET,METHOD, F)

  return n2v_scores, train_edge_labels,test_edge_labels, test_preds,emb_matrix,train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs,train_embs_1,train_embs_2,test_embs_1,test_embs_2



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

            # #with open('/Users/xiulingwang/Downloads/line-master/data/embds/' + str(ego_user) + flag + '-' + Flag, 'w') as f:
            # with open('E:/python/banlance/code/' + DATASET + '/' + METHOD + '/embeds/' + Flag + '-' + str(ego_user) + flag,'w') as f:
            # #with open('/Users/xiulingwang/Downloads/'+DATASET+'/line/3-split/' + str(ego_user)+ '-' + flag + '-' + Flag+'-' +'embds','w') as f:
            #     f.write('%d %d\n' % (edge_list.shape[0], args.representation_size))
            #     for i in range(edge_list.shape[0]):
            #         e = ' '.join(map(lambda x: str(x), embs[i]))
            #         f.write('%s %s %s\n' % (str(edge_list[i][0]), str(edge_list[i][1]), e))

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



def linkpre_scores_inf_debias(args, emb_matrix, train_test_split, ego_user,DATASET,METHOD, Flag):
    adj_train, train_edges, train_edges_false, test_edges, test_edges_false = train_test_split
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

            # #with open('/Users/xiulingwang/Downloads/line-master/data/embds/' + str(ego_user) + flag + '-' + Flag, 'w') as f:
            # with open('E:/python/banlance/code/' + DATASET + '/' + METHOD + '/embeds/' + Flag + '-' + str(ego_user) + flag,'w') as f:
            # #with open('/Users/xiulingwang/Downloads/'+DATASET+'/line/3-split/' + str(ego_user)+ '-' + flag + '-' + Flag+'-' +'embds','w') as f:
            #     f.write('%d %d\n' % (edge_list.shape[0], args.representation_size))
            #     for i in range(edge_list.shape[0]):
            #         e = ' '.join(map(lambda x: str(x), embs[i]))
            #         f.write('%s %s %s\n' % (str(edge_list[i][0]), str(edge_list[i][1]), e))

            return embs

        # Train-set edge embeddings
        pos_train_edge_embs = get_edge_embeddings(train_edges,ego_user,DATASET, Flag, flag='pos-train')
        neg_train_edge_embs = get_edge_embeddings(train_edges_false, ego_user,DATASET,Flag, flag='neg-train')
        train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

        # Create train-set edge labels: 1 = real edge, 0 = false edge
        train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

        # # Val-set edge embeddings, labels
        # if len(val_edges) > 0 and len(val_edges_false) > 0:
        #     pos_val_edge_embs = get_edge_embeddings(val_edges,ego_user,DATASET,Flag, flag='pos-val')
        #     neg_val_edge_embs = get_edge_embeddings(val_edges_false,ego_user,DATASET,Flag, flag='neg-val')
        #     val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
        #     val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

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
        # if len(val_edges) > 0 and len(val_edges_false) > 0:
        #     val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
        test_preds = edge_classifier.predict_proba(test_edge_embs)
        print(test_preds)
        print(np.shape(test_preds))

        runtime = time.time() - start_time

        # # Calculate scores
        # if len(val_edges) > 0 and len(val_edges_false) > 0:
        #     n2v_val_roc = roc_auc_score(val_edge_labels, val_preds)
        #     # n2v_val_roc_curve = roc_curve(val_edge_labels, val_preds)
        #     n2v_val_ap = average_precision_score(val_edge_labels, val_preds)
        # else:
        #     n2v_val_roc = None
        #     n2v_val_roc_curve = None
        #     n2v_val_ap = None

        # n2v_test_roc = roc_auc_score(test_edge_labels, test_preds)
        # # n2v_test_roc_curve = roc_curve(test_edge_labels, test_preds)
        # n2v_test_ap = average_precision_score(test_edge_labels, test_preds)


    # Generate edge scores using simple dot product of node embeddings (like in GAE paper)
    elif args.edge_score_mode == "dot-product":
        score_matrix = np.dot(emb_matrix, emb_matrix.T)
        runtime = time.time() - start_time

        # Val set scores
        # if len(val_edges) > 0:
        #     n2v_val_roc, n2v_val_ap = get_roc_score(val_edges, val_edges_false, score_matrix, apply_sigmoid=True)
        # else:
        #     n2v_val_roc = None
        #     n2v_val_roc_curve = None
        #     n2v_val_ap = None

        # Test set scores
        n2v_test_roc, n2v_test_ap = get_roc_score(test_edges, test_edges_false, score_matrix, apply_sigmoid=True)

    else:
        print
        "Invalid edge_score_mode! Either use edge-emb or dot-product."


    return  test_edge_labels,test_preds

    # Record scores
    # n2v_scores = {}
    #
    # n2v_scores['test_roc'] = n2v_test_roc
    # # n2v_scores['test_roc_curve'] = n2v_test_roc_curve
    # n2v_scores['test_ap'] = n2v_test_ap
    #
    # n2v_scores['val_roc'] = n2v_val_roc
    # # n2v_scores['val_roc_curve'] = n2v_val_roc_curve
    # n2v_scores['val_ap'] = n2v_val_ap
    #
    # n2v_scores['runtime'] = runtime
    #
    # return n2v_scores, val_edge_labels, val_preds, test_edge_labels, test_preds







def linkpre_scores_517(emb_matrix,train_edges,train_edges_false,test_edges,test_edges_false):
    # adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    # test_edges, test_edges_false = train_test_split
    # start_time = time.time()
    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
    # if args.edge_score_mode == "edge-emb":

    def get_edge_embeddings(edge_list):
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

        return embs

    # Train-set edge embeddings
    pos_train_edge_embs = get_edge_embeddings(train_edges)
    neg_train_edge_embs = get_edge_embeddings(train_edges_false)
    train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

    # Create train-set edge labels: 1 = real edge, 0 = false edge
    train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

    # # Val-set edge embeddings, labels
    # if len(val_edges) > 0 and len(val_edges_false) > 0:
    #     pos_val_edge_embs = get_edge_embeddings(val_edges,ego_user,DATASET,Flag, flag='pos-val')
    #     neg_val_edge_embs = get_edge_embeddings(val_edges_false,ego_user,DATASET,Flag, flag='neg-val')
    #     val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
    #     val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

    # Test-set edge embeddings, labels
    pos_test_edge_embs = get_edge_embeddings(test_edges)
    neg_test_edge_embs = get_edge_embeddings(test_edges_false)
    test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

    # Create val-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

    # Train logistic regression classifier on train-set edge embeddings
    edge_classifier = LogisticRegression(random_state=0)
    edge_classifier.fit(train_edge_embs, train_edge_labels)

    # Predicted edge scores: probability of being of class "1" (real edge)
    # if len(val_edges) > 0 and len(val_edges_false) > 0:
    #     val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
    # test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]

    test_preds = edge_classifier.predict_proba(test_edge_embs)
    # print(test_preds)
    # print(np.shape(test_preds))

    # runtime = time.time() - start_time
    #
    # Calculate scores
    # if len(val_edges) > 0 and len(val_edges_false) > 0:
    #     n2v_val_roc = roc_auc_score(val_edge_labels, val_preds)
    #     # n2v_val_roc_curve = roc_curve(val_edge_labels, val_preds)
    #     n2v_val_ap = average_precision_score(val_edge_labels, val_preds)
    # else:
    #     n2v_val_roc = None
    #     n2v_val_roc_curve = None
    #     n2v_val_ap = None

    # n2v_test_roc = roc_auc_score(test_edge_labels, test_preds)
    # # n2v_test_roc_curve = roc_curve(test_edge_labels, test_preds)
    # n2v_test_ap = average_precision_score(test_edge_labels, test_preds)


    # Generate edge scores using simple dot product of node embeddings (like in GAE paper)
    # elif args.edge_score_mode == "dot-product":
    #     score_matrix = np.dot(emb_matrix, emb_matrix.T)
    #     runtime = time.time() - start_time
    #
    #     # Val set scores
    #     if len(val_edges) > 0:
    #         n2v_val_roc, n2v_val_ap = get_roc_score(val_edges, val_edges_false, score_matrix, apply_sigmoid=True)
    #     else:
    #         n2v_val_roc = None
    #         n2v_val_roc_curve = None
    #         n2v_val_ap = None
    #
    #     # Test set scores
    #     n2v_test_roc, n2v_test_ap = get_roc_score(test_edges, test_edges_false, score_matrix, apply_sigmoid=True)
    #
    # else:
    #     print
    #     "Invalid edge_score_mode! Either use edge-emb or dot-product."

    # Record scores
    # n2v_scores = {}

    # n2v_scores['test_roc'] = n2v_test_roc
    # # n2v_scores['test_roc_curve'] = n2v_test_roc_curve
    # n2v_scores['test_ap'] = n2v_test_ap
    #
    # n2v_scores['val_roc'] = n2v_val_roc
    # # n2v_scores['val_roc_curve'] = n2v_val_roc_curve
    # n2v_scores['val_ap'] = n2v_val_ap
    #
    # n2v_scores['runtime'] = runtime

    return test_edge_labels, test_preds









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


def distCosine(x, y):
    """
    :param x: m x k array
    :param y: n x k array
    :return: m x n array
    """
    xx = np.sum(x ** 2, axis=1) ** 0.5
    x = x / xx[:, np.newaxis]
    yy = np.sum(y ** 2, axis=1) ** 0.5
    y = y / yy[:, np.newaxis]
    sim_matrix=np.dot(x, y.transpose())  # 余弦相似度
    return sim_matrix




def linkpre_scores1(args, emb_matrix, train_test_split, ego_user,DATASET,METHOD, Flag):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = train_test_split
    start_time = time.time()
    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
    if args.edge_score_mode == "edge-emb":

        def get_edge_embeddings(edge_list,ego_user,DATASET, Flag, flag):
            embs = []
            sim_matrix=[]
            embs_1=[]
            embs_2 = []
            for edge in edge_list:
                node1 = edge[0]
                node2 = edge[1]
                emb1 = emb_matrix[node1]
                #print(np.shape(emb1))
                emb2 = emb_matrix[node2]
                edge_emb = np.multiply(emb1, emb2)
                sim = np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
                #edge_emb = np.array(emb1) + np.array(emb2)
                print(np.shape(edge_emb))
                embs.append(edge_emb)
                embs_1.append(emb1)
                embs_2.append(emb2)
                sim_matrix.append(sim)
            embs = np.array(embs)
            sim_matrix = np.array(sim_matrix)
            embs_1=np.array(embs_1)
            embs_2 =np.array(embs_2)

            # #with open('/Users/xiulingwang/Downloads/line-master/data/embds/' + str(ego_user) + flag + '-' + Flag, 'w') as f:
            # with open('E:/python/banlance/code/' + DATASET + '/' + METHOD + '/embeds/' + Flag + '-' + str(ego_user) + flag,'w') as f:
            # #with open('/Users/xiulingwang/Downloads/'+DATASET+'/line/3-split/' + str(ego_user)+ '-' + flag + '-' + Flag+'-' +'embds','w') as f:
            #     f.write('%d %d\n' % (edge_list.shape[0], args.representation_size))
            #     for i in range(edge_list.shape[0]):
            #         e = ' '.join(map(lambda x: str(x), embs[i]))
            #         f.write('%s %s %s\n' % (str(edge_list[i][0]), str(edge_list[i][1]), e))

            return embs,sim_matrix,embs_1,embs_2

        # Train-set edge embeddings
        pos_train_edge_embs ,pos_train_sim_matrix,pos_embs_1_train,pos_embs_2_train= get_edge_embeddings(train_edges,ego_user,DATASET, Flag, flag='pos-train')
        neg_train_edge_embs,neg_train_sim_matrix,neg_embs_1_train,neg_embs_2_train = get_edge_embeddings(train_edges_false, ego_user,DATASET,Flag, flag='neg-train')
        train_edge_embs = np.concatenate((pos_train_edge_embs, neg_train_edge_embs), axis=0)
        train_sim_matrix= np.concatenate((pos_train_sim_matrix, neg_train_sim_matrix), axis=0)
        train_embs_1 = np.concatenate((pos_embs_1_train, neg_embs_1_train), axis=0)
        train_embs_2 = np.concatenate((pos_embs_2_train, neg_embs_2_train), axis=0)

        # Create train-set edge labels: 1 = real edge, 0 = false edge
        train_edge_labels = np.concatenate((np.ones(len(train_edges)), np.zeros(len(train_edges_false))), axis=0)

        # Val-set edge embeddings, labels
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            pos_val_edge_embs,pos_val_sim_matrix,pos_embs_1_val,pos_embs_2_val = get_edge_embeddings(val_edges,ego_user,DATASET,Flag, flag='pos-val')
            neg_val_edge_embs,neg_val_sim_matrix,neg_embs_1_val,neg_embs_2_val = get_edge_embeddings(val_edges_false,ego_user,DATASET,Flag, flag='neg-val')
            val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
            val_edge_labels = np.concatenate((np.ones(len(val_edges)), np.zeros(len(val_edges_false))), axis=0)
            val_sim_matrix = np.concatenate((pos_val_sim_matrix, neg_val_sim_matrix), axis=0)
            val_embs_1 = np.concatenate((pos_embs_1_val, neg_embs_1_val), axis=0)
            val_embs_2 = np.concatenate((pos_embs_2_val, neg_embs_2_val), axis=0)

        # Test-set edge embeddings, labels
        pos_test_edge_embs,pos_test_sim_matrix,pos_embs_1_test,pos_embs_2_test = get_edge_embeddings(test_edges,ego_user,DATASET,Flag, flag='pos-test')
        neg_test_edge_embs ,neg_test_sim_matrix,neg_embs_1_test,neg_embs_2_test= get_edge_embeddings(test_edges_false,ego_user,DATASET,Flag, flag='neg-test')
        test_edge_embs = np.concatenate((pos_test_edge_embs, neg_test_edge_embs), axis=0)
        test_sim_matrix = np.concatenate((pos_test_sim_matrix, neg_test_sim_matrix), axis=0)
        test_embs_1 = np.concatenate((pos_embs_1_test, neg_embs_1_test), axis=0)
        test_embs_2 = np.concatenate((pos_embs_2_test, neg_embs_2_test), axis=0)

        # Create val-set edge labels: 1 = real edge, 0 = false edge
        test_edge_labels = np.concatenate((np.ones(len(test_edges)), np.zeros(len(test_edges_false))), axis=0)

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

    return n2v_scores, train_edge_labels,test_edge_labels, test_preds,train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs,train_embs_1,train_embs_2,test_embs_1,test_embs_2



def deepwalk2(g_train, train_test_split,DATASET,METHOD,res_dir, ego_user, F,dp):
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
  args.input='%s%s-fair-%s-train.txt' % (res_dir, str(ego_user), F)
  args.output = res_dir+METHOD+'-embeds-'+F+'-'+str(ego_user)
  numeric_level = getattr(logging, args.log.upper(), None)
  logging.basicConfig(format=LOGFORMAT)
  logger.setLevel(numeric_level)

  if args.debug:
   sys.excepthook = debug

  adj_train_orig, train_edges, train_edges_false, val_edges, val_edges_false, \
  test_edges, test_edges_false = train_test_split
  g_train=nx.adjacency_matrix(g_train)
  emb_matrix=process3(args,g_train,DATASET,METHOD,F,ego_user,res_dir,dp)

  n2v_scores,train_edge_labels, test_edge_labels, test_preds,train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs,train_embs_1,train_embs_2,test_embs_1,test_embs_2=linkpre_scores2(args, emb_matrix, train_test_split, ego_user,DATASET,METHOD, F)

  return n2v_scores, train_edge_labels,test_edge_labels, test_preds,emb_matrix,train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs,train_embs_1,train_embs_2,test_embs_1,test_embs_2



def linkpre_scores2(args, emb_matrix, train_test_split, ego_user,DATASET,METHOD, Flag):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = train_test_split
    start_time = time.time()
    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
    if args.edge_score_mode == "edge-emb":

        def get_edge_embeddings(edge_list,ego_user,DATASET, Flag, flag):
            embs = []
            sim_matrix=[]
            embs_1=[]
            embs_2 = []
            for edge in edge_list:
                node1 = edge[0]
                node2 = edge[1]
                emb1 = emb_matrix[node1]
                #print(np.shape(emb1))
                emb2 = emb_matrix[node2]
                edge_emb = np.multiply(emb1, emb2)
                sim = np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
                #edge_emb = np.array(emb1) + np.array(emb2)
                #print(np.shape(edge_emb))
                embs.append(edge_emb)
                embs_1.append(emb1)
                embs_2.append(emb2)
                sim_matrix.append(sim)
            embs = np.array(embs)
            sim_matrix = np.array(sim_matrix)
            embs_1=np.array(embs_1)
            embs_2 =np.array(embs_2)

            # #with open('/Users/xiulingwang/Downloads/line-master/data/embds/' + str(ego_user) + flag + '-' + Flag, 'w') as f:
            # with open('E:/python/banlance/code/' + DATASET + '/' + METHOD + '/embeds/' + Flag + '-' + str(ego_user) + flag,'w') as f:
            # #with open('/Users/xiulingwang/Downloads/'+DATASET+'/line/3-split/' + str(ego_user)+ '-' + flag + '-' + Flag+'-' +'embds','w') as f:
            #     f.write('%d %d\n' % (edge_list.shape[0], args.representation_size))
            #     for i in range(edge_list.shape[0]):
            #         e = ' '.join(map(lambda x: str(x), embs[i]))
            #         f.write('%s %s %s\n' % (str(edge_list[i][0]), str(edge_list[i][1]), e))

            return embs,sim_matrix,embs_1,embs_2

        # Train-set edge embeddings
        pos_train_edge_embs ,pos_train_sim_matrix,pos_embs_1_train,pos_embs_2_train= get_edge_embeddings(train_edges,ego_user,DATASET, Flag, flag='pos-train')
        neg_train_edge_embs,neg_train_sim_matrix,neg_embs_1_train,neg_embs_2_train = get_edge_embeddings(train_edges_false, ego_user,DATASET,Flag, flag='neg-train')
        train_edge_embs = np.concatenate((pos_train_edge_embs, neg_train_edge_embs), axis=0)
        train_sim_matrix= np.concatenate((pos_train_sim_matrix, neg_train_sim_matrix), axis=0)
        train_embs_1 = np.concatenate((pos_embs_1_train, neg_embs_1_train), axis=0)
        train_embs_2 = np.concatenate((pos_embs_2_train, neg_embs_2_train), axis=0)

        # Create train-set edge labels: 1 = real edge, 0 = false edge
        train_edge_labels = np.concatenate((np.ones(len(train_edges)), np.zeros(len(train_edges_false))), axis=0)

        # Val-set edge embeddings, labels
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            pos_val_edge_embs,pos_val_sim_matrix,pos_embs_1_val,pos_embs_2_val = get_edge_embeddings(val_edges,ego_user,DATASET,Flag, flag='pos-val')
            neg_val_edge_embs,neg_val_sim_matrix,neg_embs_1_val,neg_embs_2_val = get_edge_embeddings(val_edges_false,ego_user,DATASET,Flag, flag='neg-val')
            val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
            val_edge_labels = np.concatenate((np.ones(len(val_edges)), np.zeros(len(val_edges_false))), axis=0)
            val_sim_matrix = np.concatenate((pos_val_sim_matrix, neg_val_sim_matrix), axis=0)
            val_embs_1 = np.concatenate((pos_embs_1_val, neg_embs_1_val), axis=0)
            val_embs_2 = np.concatenate((pos_embs_2_val, neg_embs_2_val), axis=0)

        # Test-set edge embeddings, labels
        pos_test_edge_embs,pos_test_sim_matrix,pos_embs_1_test,pos_embs_2_test = get_edge_embeddings(test_edges,ego_user,DATASET,Flag, flag='pos-test')
        neg_test_edge_embs ,neg_test_sim_matrix,neg_embs_1_test,neg_embs_2_test= get_edge_embeddings(test_edges_false,ego_user,DATASET,Flag, flag='neg-test')
        test_edge_embs = np.concatenate((pos_test_edge_embs, neg_test_edge_embs), axis=0)
        test_sim_matrix = np.concatenate((pos_test_sim_matrix, neg_test_sim_matrix), axis=0)
        test_embs_1 = np.concatenate((pos_embs_1_test, neg_embs_1_test), axis=0)
        test_embs_2 = np.concatenate((pos_embs_2_test, neg_embs_2_test), axis=0)

        # Create val-set edge labels: 1 = real edge, 0 = false edge
        test_edge_labels = np.concatenate((np.ones(len(test_edges)), np.zeros(len(test_edges_false))), axis=0)

        # Train logistic regression classifier on train-set edge embeddings
        edge_classifier = LogisticRegression(random_state=0)
        edge_classifier.fit(train_edge_embs, train_edge_labels)

        # Predicted edge scores: probability of being of class "1" (real edge)
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
        test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
        # print(test_preds)
        #print(np.shape(test_preds))

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

    return n2v_scores, train_edge_labels,test_edge_labels, test_preds,train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs,train_embs_1,train_embs_2,test_embs_1,test_embs_2


def process2(args,adj_train,DATASET,METHOD,F,ego_user,res_dir,dp):

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
    walks = graph.build_deepwalk_corpus1(G, num_paths=args.number_walks,
                                        path_length=args.walk_length, alpha=0, rand=random.Random(args.seed))

    # print(walks)
    # print(type(walks))

    write_walks2(args, walks, DATASET, METHOD, F, ego_user,res_dir)
    print("Training...")

  class Model(torch.nn.Module):
      def __init__(self):
          super(Model, self).__init__()
          self.phi = nn.Parameter(torch.rand((len(G.nodes()), args.representation_size), requires_grad=True))
          self.phi2 = nn.Parameter(torch.rand((args.representation_size, len(G.nodes())), requires_grad=True))

      def forward(self, one_hot):
          hidden = torch.matmul(one_hot, self.phi)
          out = torch.matmul(hidden, self.phi2)
          return out

  model = Model()

  def skip_gram(wvi, w):
      for j in range(len(wvi)):
          for k in range(max(0, j - w), min(j + w, len(wvi))):

              # generate one hot vector
              one_hot = torch.zeros(len(G.nodes()))
              one_hot[wvi[j]] = 1

              out = model(one_hot)
              loss = torch.log(torch.sum(torch.exp(out))) - out[wvi[k]]
              loss.backward()
              lr=0.025
              for param in model.parameters():
                  param.data.sub_(lr * param.grad)
                  param.grad.data.zero_()



  def skip_gram_dp(wvi, w):
      C=1
      sigma=4
      for j in range(len(wvi)):
          for k in range(max(0, j - w), min(j + w, len(wvi))):



              # generate one hot vector
              one_hot = torch.zeros(len(G.nodes()))
              one_hot[wvi[j]] = 1

              out = model(one_hot)
              loss = torch.log(torch.sum(torch.exp(out))) - out[wvi[k]]

              loss.backward()

              l2_norm = torch.tensor(0.0)
              for param in model.parameters():
                  temp_grad=param.grad
                  l2_norm += temp_grad.norm(2) ** 2

              l2_norm = l2_norm.sqrt()
              divisor = max(torch.tensor(1.0), l2_norm / C)

              for param in model.parameters():
                  param.grad += param.grad / divisor

                  param.grad += sigma * C * (torch.randn_like(param.grad))

                  print(param.grad)



              lr=0.025
              for param in model.parameters():
                  param.data.sub_(lr * param.grad)
                  param.grad.data.zero_()


  if (dp==1):
      for walk in walks:
          # print(walk)
          skip_gram_dp(walk, args.window_size)

  if (dp==0):
      for walk in walks:
          # print(walk)
          skip_gram(walk, args.window_size)

  emb_mappings=model.phi
  emb_mappings=emb_mappings.detach().numpy()
  print(np.shape(emb_mappings))

  #   model = Word2Vec(walks, size=args.representation_size, window=args.window_size, min_count=0, sg=1, hs=1, workers=args.workers)
  #
  #   training_loss = model.get_latest_training_loss()
  #
  # else:
  #   print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size, args.max_memory_data_size))
  #   print("Walking...")
  #
  #   walks_filebase = args.output + ".walks"
  #   walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=args.number_walks,
  #                                        path_length=args.walk_length, alpha=0, rand=random.Random(args.seed),
  #                                        num_workers=args.workers)
  #
  #   print("Counting vertex frequency...")
  #   if not args.vertex_freq_degree:
  #     vertex_counts = serialized_walks.count_textfiles(walk_files, args.workers)
  #   else:
  #     # use degree distribution for frequency in tree
  #     vertex_counts = G.degree(nodes=G.iterkeys())
  #
  #   print("Training...")
  #   walks_corpus = serialized_walks.WalksCorpus(walk_files)
  #   model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
  #                    size=args.representation_size,
  #                    window=args.window_size, min_count=0, trim_rule=None, workers=args.workers)
  #
  # model.wv.save_word2vec_format(args.output)
  #
  # exit()



  # Store embeddings mapping
  #emb_mappings = model.wv

  # Create node embeddings matrix (rows = nodes, columns = embedding features)
  emb_list = []
  for node_index in range(0, adj_train.shape[0]):
      node_str = int(node_index)
      node_emb = emb_mappings[node_str]
      emb_list.append(node_emb)
  emb_matrix = np.vstack(emb_list)

  with open(res_dir + DATASET + '-' + METHOD + '-embeds-' + F + '-' + str(ego_user), 'w') as f:
      f.write('%d %d\n' % (adj_train.shape[0], args.representation_size))
      for i in range(adj_train.shape[0]):
          e = ' '.join(map(lambda x: str(x), emb_list[i]))
          f.write('%s %s\n' % (str(i), e))

  return emb_matrix



def process3(args,adj_train,DATASET,METHOD,F,ego_user,res_dir,dp):

  if args.format == "adjlist":
    G = graph.load_adjacencylist(args.input, undirected=args.undirected)
  elif args.format == "edgelist":
    print('&&&&&')
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
    walks = graph.build_deepwalk_corpus1(G, num_paths=args.number_walks,
                                        path_length=args.walk_length, alpha=0, rand=random.Random(args.seed))

    # print(walks)
    # print(type(walks))ee4

    write_walks2(args, walks, DATASET, METHOD, F, ego_user,res_dir)
    print("Training...")

    # model = Word2Vec(walks, size=args.representation_size, window=args.window_size, min_count=0, sg=1, hs=1,
    #                  workers=args.workers)

    input_file=res_dir + DATASET + '-' + METHOD + '-walks-' + F + '-' + str(ego_user)

    w2v = trainer.Word2VecTrainer(input_file, output_file="out.vec")
    if dp==0:
        emb_mappings=w2v.train(res_dir,DATASET,METHOD, F,ego_user)
    if dp==1:
        emb_mappings=w2v.train_dp(res_dir,DATASET,METHOD, F,ego_user)


  #emb_mappings=w2v.u_embeddings.weight.cpu().data.numpy()
  emb_mappings=emb_mappings.cpu().detach().numpy()
  # print(np.shape(emb_mappings))

  #   model = Word2Vec(walks, size=args.representation_size, window=args.window_size, min_count=0, sg=1, hs=1, workers=args.workers)
  #
  #   training_loss = model.get_latest_training_loss()
  #
  # else:
  #   print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size, args.max_memory_data_size))
  #   print("Walking...")
  #
  #   walks_filebase = args.output + ".walks"
  #   walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=args.number_walks,
  #                                        path_length=args.walk_length, alpha=0, rand=random.Random(args.seed),
  #                                        num_workers=args.workers)
  #
  #   print("Counting vertex frequency...")
  #   if not args.vertex_freq_degree:
  #     vertex_counts = serialized_walks.count_textfiles(walk_files, args.workers)
  #   else:
  #     # use degree distribution for frequency in tree
  #     vertex_counts = G.degree(nodes=G.iterkeys())
  #
  #   print("Training...")
  #   walks_corpus = serialized_walks.WalksCorpus(walk_files)
  #   model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
  #                    size=args.representation_size,
  #                    window=args.window_size, min_count=0, trim_rule=None, workers=args.workers)
  #
  # model.wv.save_word2vec_format(args.output)
  #
  # exit()



  # Store embeddings mapping
  #emb_mappings = model.wv

  # Create node embeddings matrix (rows = nodes, columns = embedding features)
  emb_list = []
  for node_index in range(0, adj_train.shape[0]):
      node_str = int(node_index)
      node_emb = emb_mappings[node_str]
      emb_list.append(node_emb)
  emb_matrix = np.vstack(emb_list)

  with open(res_dir + DATASET + '-' + METHOD + '-embeds-' + F + '-' + str(ego_user), 'w') as f:
      f.write('%d %d\n' % (adj_train.shape[0], args.representation_size))
      for i in range(adj_train.shape[0]):
          e = ' '.join(map(lambda x: str(x), emb_list[i]))
          f.write('%s %s\n' % (str(i), e))

  return emb_matrix


def deepwalk3(g_train, train_test_split,other_edge,DATASET,METHOD,res_dir, ego_user, F,dp):
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
  args.input='%s%s-fair-%s-train.txt' % (res_dir, str(ego_user), F)
  args.output = res_dir+METHOD+'-embeds-'+F+'-'+str(ego_user)
  numeric_level = getattr(logging, args.log.upper(), None)
  logging.basicConfig(format=LOGFORMAT)
  logger.setLevel(numeric_level)

  if args.debug:
   sys.excepthook = debug

  adj_train_orig, train_edges, train_edges_false, val_edges, val_edges_false, \
  test_edges, test_edges_false = train_test_split
  g_train=nx.adjacency_matrix(g_train)
  emb_matrix=process3(args,g_train,DATASET,METHOD,F,ego_user,res_dir,dp)

  n2v_scores,train_edge_labels, test_edge_labels, test_preds,train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs,train_embs_1,train_embs_2,test_embs_1,test_embs_2, other_edge_embs, other_sim_matrix, other_embs_1_test, other_embs_2_test, other_edge_labels=linkpre_scores3(args, emb_matrix, train_test_split, other_edge,ego_user,DATASET,METHOD, F)

  return n2v_scores, train_edge_labels,test_edge_labels, test_preds,emb_matrix,train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs,train_embs_1,train_embs_2,test_embs_1,test_embs_2, other_edge_embs, other_sim_matrix, other_embs_1_test, other_embs_2_test, other_edge_labels

def linkpre_scores3(args, emb_matrix, train_test_split, other_edge,ego_user,DATASET,METHOD, Flag):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = train_test_split
    start_time = time.time()
    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
    if args.edge_score_mode == "edge-emb":

        def get_edge_embeddings(edge_list,ego_user,DATASET, Flag, flag):
            embs = []
            sim_matrix=[]
            embs_1=[]
            embs_2 = []
            for edge in edge_list:
                node1 = edge[0]
                node2 = edge[1]
                emb1 = emb_matrix[node1]
                #print(np.shape(emb1))
                emb2 = emb_matrix[node2]
                edge_emb = np.multiply(emb1, emb2)
                sim = np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
                #edge_emb = np.array(emb1) + np.array(emb2)
                #print(np.shape(edge_emb))
                embs.append(edge_emb)
                embs_1.append(emb1)
                embs_2.append(emb2)
                sim_matrix.append(sim)
            embs = np.array(embs)
            sim_matrix = np.array(sim_matrix)
            embs_1=np.array(embs_1)
            embs_2 =np.array(embs_2)

            # #with open('/Users/xiulingwang/Downloads/line-master/data/embds/' + str(ego_user) + flag + '-' + Flag, 'w') as f:
            # with open('E:/python/banlance/code/' + DATASET + '/' + METHOD + '/embeds/' + Flag + '-' + str(ego_user) + flag,'w') as f:
            # #with open('/Users/xiulingwang/Downloads/'+DATASET+'/line/3-split/' + str(ego_user)+ '-' + flag + '-' + Flag+'-' +'embds','w') as f:
            #     f.write('%d %d\n' % (edge_list.shape[0], args.representation_size))
            #     for i in range(edge_list.shape[0]):
            #         e = ' '.join(map(lambda x: str(x), embs[i]))
            #         f.write('%s %s %s\n' % (str(edge_list[i][0]), str(edge_list[i][1]), e))

            return embs,sim_matrix,embs_1,embs_2

        # Train-set edge embeddings
        pos_train_edge_embs ,pos_train_sim_matrix,pos_embs_1_train,pos_embs_2_train= get_edge_embeddings(train_edges,ego_user,DATASET, Flag, flag='pos-train')
        neg_train_edge_embs,neg_train_sim_matrix,neg_embs_1_train,neg_embs_2_train = get_edge_embeddings(train_edges_false, ego_user,DATASET,Flag, flag='neg-train')
        train_edge_embs = np.concatenate((pos_train_edge_embs, neg_train_edge_embs), axis=0)
        train_sim_matrix= np.concatenate((pos_train_sim_matrix, neg_train_sim_matrix), axis=0)
        train_embs_1 = np.concatenate((pos_embs_1_train, neg_embs_1_train), axis=0)
        train_embs_2 = np.concatenate((pos_embs_2_train, neg_embs_2_train), axis=0)

        # Create train-set edge labels: 1 = real edge, 0 = false edge
        train_edge_labels = np.concatenate((np.ones(len(train_edges)), np.zeros(len(train_edges_false))), axis=0)

        # Val-set edge embeddings, labels
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            pos_val_edge_embs,pos_val_sim_matrix,pos_embs_1_val,pos_embs_2_val = get_edge_embeddings(val_edges,ego_user,DATASET,Flag, flag='pos-val')
            neg_val_edge_embs,neg_val_sim_matrix,neg_embs_1_val,neg_embs_2_val = get_edge_embeddings(val_edges_false,ego_user,DATASET,Flag, flag='neg-val')
            val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
            val_edge_labels = np.concatenate((np.ones(len(val_edges)), np.zeros(len(val_edges_false))), axis=0)
            val_sim_matrix = np.concatenate((pos_val_sim_matrix, neg_val_sim_matrix), axis=0)
            val_embs_1 = np.concatenate((pos_embs_1_val, neg_embs_1_val), axis=0)
            val_embs_2 = np.concatenate((pos_embs_2_val, neg_embs_2_val), axis=0)

        # Test-set edge embeddings, labels
        pos_test_edge_embs,pos_test_sim_matrix,pos_embs_1_test,pos_embs_2_test = get_edge_embeddings(test_edges,ego_user,DATASET,Flag, flag='pos-test')
        neg_test_edge_embs ,neg_test_sim_matrix,neg_embs_1_test,neg_embs_2_test= get_edge_embeddings(test_edges_false,ego_user,DATASET,Flag, flag='neg-test')

        test_edge_embs = np.concatenate((pos_test_edge_embs, neg_test_edge_embs), axis=0)
        test_sim_matrix = np.concatenate((pos_test_sim_matrix, neg_test_sim_matrix), axis=0)
        test_embs_1 = np.concatenate((pos_embs_1_test, neg_embs_1_test), axis=0)
        test_embs_2 = np.concatenate((pos_embs_2_test, neg_embs_2_test), axis=0)

        # Create val-set edge labels: 1 = real edge, 0 = false edge
        test_edge_labels = np.concatenate((np.ones(len(test_edges)), np.zeros(len(test_edges_false))), axis=0)

        other_edge_embs, other_sim_matrix, other_embs_1_test, other_embs_2_test = get_edge_embeddings(
            other_edge, ego_user, DATASET, Flag, flag='other')
        other_edge_labels = np.zeros(len(other_edge))

        # Train logistic regression classifier on train-set edge embeddings
        edge_classifier = LogisticRegression(random_state=0)
        edge_classifier.fit(train_edge_embs, train_edge_labels)

        # Predicted edge scores: probability of being of class "1" (real edge)
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
        test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
        # print(test_preds)
        #print(np.shape(test_preds))

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

    return n2v_scores, train_edge_labels,test_edge_labels, test_preds,train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs,train_embs_1,train_embs_2,test_embs_1,test_embs_2, other_edge_embs, other_sim_matrix, other_embs_1_test, other_embs_2_test, other_edge_labels

def deepwalk4(g_train, train_test_split,DATASET,METHOD,res_dir, ego_user, F,dp):
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
  args.input='%s%s-fair-%s-train.txt' % (res_dir, str(ego_user), F)
  args.output = res_dir+METHOD+'-embeds-'+F+'-'+str(ego_user)
  numeric_level = getattr(logging, args.log.upper(), None)
  logging.basicConfig(format=LOGFORMAT)
  logger.setLevel(numeric_level)

  if args.debug:
   sys.excepthook = debug

  adj_train_orig, train_edges, train_edges_false, val_edges, val_edges_false, \
  test_edges, test_edges_false = train_test_split
  g_train=nx.adjacency_matrix(g_train)
  emb_matrix=process3(args,g_train,DATASET,METHOD,F,ego_user,res_dir,dp)

  # n2v_scores,train_edge_labels, test_edge_labels, test_preds,train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs,train_embs_1,train_embs_2,test_embs_1,test_embs_2, other_edge_embs, other_sim_matrix, other_embs_1_test, other_embs_2_test, other_edge_labels=linkpre_scores4(args, emb_matrix, train_test_split, ego_user,DATASET,METHOD, F)

  return emb_matrix

def linkpre_scores4(emb_matrix, train_edges_pos,train_edges_neg,test_edges, other_edge):

    start_time = time.time()
    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2

    def get_edge_embeddings(edge_list):
        embs = []
        sim_matrix=[]
        embs_1=[]
        embs_2 = []
        for edge in edge_list:
            node1 = edge[0]
            node2 = edge[1]
            emb1 = emb_matrix[node1]
            #print(np.shape(emb1))
            emb2 = emb_matrix[node2]
            edge_emb = np.multiply(emb1, emb2)
            sim = np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
            #edge_emb = np.array(emb1) + np.array(emb2)
            #print(np.shape(edge_emb))
            embs.append(edge_emb)
            embs_1.append(emb1)
            embs_2.append(emb2)
            sim_matrix.append(sim)
        embs = np.array(embs)
        sim_matrix = np.array(sim_matrix)
        embs_1=np.array(embs_1)
        embs_2 =np.array(embs_2)

        # #with open('/Users/xiulingwang/Downloads/line-master/data/embds/' + str(ego_user) + flag + '-' + Flag, 'w') as f:
        # with open('E:/python/banlance/code/' + DATASET + '/' + METHOD + '/embeds/' + Flag + '-' + str(ego_user) + flag,'w') as f:
        # #with open('/Users/xiulingwang/Downloads/'+DATASET+'/line/3-split/' + str(ego_user)+ '-' + flag + '-' + Flag+'-' +'embds','w') as f:
        #     f.write('%d %d\n' % (edge_list.shape[0], args.representation_size))
        #     for i in range(edge_list.shape[0]):
        #         e = ' '.join(map(lambda x: str(x), embs[i]))
        #         f.write('%s %s %s\n' % (str(edge_list[i][0]), str(edge_list[i][1]), e))

        return embs,sim_matrix,embs_1,embs_2

    # Train-set edge embeddings
    pos_train_edge_embs ,pos_train_sim_matrix,pos_embs_1_train,pos_embs_2_train= get_edge_embeddings(train_edges_pos)
    neg_train_edge_embs,neg_train_sim_matrix,neg_embs_1_train,neg_embs_2_train = get_edge_embeddings(train_edges_neg)
    train_edge_embs = np.concatenate((pos_train_edge_embs, neg_train_edge_embs), axis=0)
    train_sim_matrix= np.concatenate((pos_train_sim_matrix, neg_train_sim_matrix), axis=0)
    train_embs_1 = np.concatenate((pos_embs_1_train, neg_embs_1_train), axis=0)
    train_embs_2 = np.concatenate((pos_embs_2_train, neg_embs_2_train), axis=0)

    # Create train-set edge labels: 1 = real edge, 0 = false edge
    train_edge_labels = np.concatenate((np.ones(len(train_edges_pos)), np.zeros(len(train_edges_neg))), axis=0)


    # Test-set edge embeddings, labels
    pos_test_edge_embs,pos_test_sim_matrix,pos_embs_1_test,pos_embs_2_test = get_edge_embeddings(test_edges)
    neg_test_edge_embs ,neg_test_sim_matrix,neg_embs_1_test,neg_embs_2_test= get_edge_embeddings(other_edge)

    test_edge_embs = np.concatenate((pos_test_edge_embs, neg_test_edge_embs), axis=0)
    test_sim_matrix = np.concatenate((pos_test_sim_matrix, neg_test_sim_matrix), axis=0)
    test_embs_1 = np.concatenate((pos_embs_1_test, neg_embs_1_test), axis=0)
    test_embs_2 = np.concatenate((pos_embs_2_test, neg_embs_2_test), axis=0)

    # Create val-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.concatenate((np.ones(len(test_edges)), np.zeros(len(other_edge))), axis=0)

    # Train logistic regression classifier on train-set edge embeddings
    edge_classifier = LogisticRegression(random_state=0)
    edge_classifier.fit(train_edge_embs, train_edge_labels)

    # Predicted edge scores: probability of being of class "1" (real edge)

    test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
    # print(test_preds)
    #print(np.shape(test_preds))

    runtime = time.time() - start_time

    # Calculate scores

    n2v_test_roc = roc_auc_score(test_edge_labels, test_preds)
    # n2v_test_roc_curve = roc_curve(test_edge_labels, test_preds)
    n2v_test_ap = average_precision_score(test_edge_labels, test_preds)


    # Record scores
    n2v_scores = {}

    n2v_scores['test_roc'] = n2v_test_roc
    # n2v_scores['test_roc_curve'] = n2v_test_roc_curve
    n2v_scores['test_ap'] = n2v_test_ap

    n2v_scores['runtime'] = runtime

    return n2v_scores, train_edge_labels,test_edge_labels, test_preds,train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs,train_embs_1,train_embs_2,test_embs_1,test_embs_2


def linkpre_scores5(emb_matrix, train_edges,test_edges):

    start_time = time.time()
    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2

    def get_edge_embeddings(edge_list):
        embs = []
        sim_matrix=[]
        embs_1=[]
        embs_2 = []
        for edge in edge_list:
            node1 = edge[0]
            node2 = edge[1]
            emb1 = emb_matrix[node1]
            #print(np.shape(emb1))
            emb2 = emb_matrix[node2]
            edge_emb = np.multiply(emb1, emb2)
            sim = np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
            #edge_emb = np.array(emb1) + np.array(emb2)
            #print(np.shape(edge_emb))
            embs.append(edge_emb)
            embs_1.append(emb1)
            embs_2.append(emb2)
            sim_matrix.append(sim)
        embs = np.array(embs)
        sim_matrix = np.array(sim_matrix)
        embs_1=np.array(embs_1)
        embs_2 =np.array(embs_2)

        # #with open('/Users/xiulingwang/Downloads/line-master/data/embds/' + str(ego_user) + flag + '-' + Flag, 'w') as f:
        # with open('E:/python/banlance/code/' + DATASET + '/' + METHOD + '/embeds/' + Flag + '-' + str(ego_user) + flag,'w') as f:
        # #with open('/Users/xiulingwang/Downloads/'+DATASET+'/line/3-split/' + str(ego_user)+ '-' + flag + '-' + Flag+'-' +'embds','w') as f:
        #     f.write('%d %d\n' % (edge_list.shape[0], args.representation_size))
        #     for i in range(edge_list.shape[0]):
        #         e = ' '.join(map(lambda x: str(x), embs[i]))
        #         f.write('%s %s %s\n' % (str(edge_list[i][0]), str(edge_list[i][1]), e))

        return embs,sim_matrix,embs_1,embs_2

    # Train-set edge embeddings
    # pos_train_edge_embs ,pos_train_sim_matrix,pos_embs_1_train,pos_embs_2_train= get_edge_embeddings(train_edges_pos)
    # neg_train_edge_embs,neg_train_sim_matrix,neg_embs_1_train,neg_embs_2_train = get_edge_embeddings(train_edges_neg)
    # train_edge_embs = np.concatenate((pos_train_edge_embs, neg_train_edge_embs), axis=0)
    # train_sim_matrix= np.concatenate((pos_train_sim_matrix, neg_train_sim_matrix), axis=0)
    # train_embs_1 = np.concatenate((pos_embs_1_train, neg_embs_1_train), axis=0)
    # train_embs_2 = np.concatenate((pos_embs_2_train, neg_embs_2_train), axis=0)

    train_edge_embs, train_sim_matrix, embs_1_train, embs_2_train = get_edge_embeddings(train_edges)

    # Create train-set edge labels: 1 = real edge, 0 = false edge
    train_edge_labels = np.ones(len(train_edges))


    # Test-set edge embeddings, labels
    test_edge_embs,test_sim_matrix,embs_1_test,embs_2_test = get_edge_embeddings(test_edges)


    # Create val-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.ones(len(test_edges))

    # Train logistic regression classifier on train-set edge embeddings
    edge_classifier = LogisticRegression(random_state=0)
    edge_classifier.fit(train_edge_embs, train_edge_labels)

    # Predicted edge scores: probability of being of class "1" (real edge)

    test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
    # print(test_preds)
    #print(np.shape(test_preds))

    runtime = time.time() - start_time

    # Calculate scores

    n2v_test_roc = roc_auc_score(test_edge_labels, test_preds)
    # n2v_test_roc_curve = roc_curve(test_edge_labels, test_preds)
    n2v_test_ap = average_precision_score(test_edge_labels, test_preds)


    # Record scores
    n2v_scores = {}

    n2v_scores['test_roc'] = n2v_test_roc
    # n2v_scores['test_roc_curve'] = n2v_test_roc_curve
    n2v_scores['test_ap'] = n2v_test_ap

    n2v_scores['runtime'] = runtime

    print(n2v_scores)

    return n2v_scores, train_edge_labels,test_edge_labels, test_preds,train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs


def deepwalk6(g_train, train_test_split,DATASET,METHOD,res_dir, ego_user, F,dp):
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
  args.input='%s%s-fair-%s-train.txt' % (res_dir, str(ego_user), F)
  args.output = res_dir+METHOD+'-embeds-'+F+'-'+str(ego_user)
  numeric_level = getattr(logging, args.log.upper(), None)
  logging.basicConfig(format=LOGFORMAT)
  logger.setLevel(numeric_level)

  if args.debug:
   sys.excepthook = debug

  adj_train_orig, train_edges, train_edges_false, val_edges, val_edges_false, \
  test_edges, test_edges_false = train_test_split
  g_train=nx.adjacency_matrix(g_train)
  emb_matrix=process3(args,g_train,DATASET,METHOD,F,ego_user,res_dir,dp)

  train_edge_labels, test_edge_labels, train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs,train_embs_1,train_embs_2,test_embs_1,test_embs_2,train_edges_sampled=linkpre_scores6(args, emb_matrix, train_test_split, ego_user,DATASET,METHOD, F)

  return  train_edge_labels,test_edge_labels, emb_matrix,train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs,train_embs_1,train_embs_2,test_embs_1,test_embs_2,train_edges_sampled



def linkpre_scores6(args, emb_matrix, train_test_split, ego_user,DATASET,METHOD, Flag):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = train_test_split
    start_time = time.time()
    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
    if args.edge_score_mode == "edge-emb":

        def get_edge_embeddings(edge_list,ego_user,DATASET, Flag, flag):
            embs = []
            sim_matrix=[]
            embs_1=[]
            embs_2 = []
            for edge in edge_list:
                node1 = edge[0]
                node2 = edge[1]
                emb1 = emb_matrix[node1]
                #print(np.shape(emb1))
                emb2 = emb_matrix[node2]
                edge_emb = np.multiply(emb1, emb2)
                # sim = np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
                sim = np.dot(emb1, emb2)
                #edge_emb = np.array(emb1) + np.array(emb2)
                #print(np.shape(edge_emb))
                embs.append(edge_emb)
                embs_1.append(emb1)
                embs_2.append(emb2)
                sim_matrix.append(sim)
            embs = np.array(embs)
            sim_matrix = np.array(sim_matrix)
            embs_1=np.array(embs_1)
            embs_2 =np.array(embs_2)

            # #with open('/Users/xiulingwang/Downloads/line-master/data/embds/' + str(ego_user) + flag + '-' + Flag, 'w') as f:
            # with open('E:/python/banlance/code/' + DATASET + '/' + METHOD + '/embeds/' + Flag + '-' + str(ego_user) + flag,'w') as f:
            # #with open('/Users/xiulingwang/Downloads/'+DATASET+'/line/3-split/' + str(ego_user)+ '-' + flag + '-' + Flag+'-' +'embds','w') as f:
            #     f.write('%d %d\n' % (edge_list.shape[0], args.representation_size))
            #     for i in range(edge_list.shape[0]):
            #         e = ' '.join(map(lambda x: str(x), embs[i]))
            #         f.write('%s %s %s\n' % (str(edge_list[i][0]), str(edge_list[i][1]), e))

            return embs,sim_matrix,embs_1,embs_2

        edgeall = list([list(edge_tuple) for edge_tuple in train_edges])

        # train_edges_sampled = random.sample(edgeall, np.shape(test_edges)[0])
        train_edges_sampled = random.sample(edgeall, np.shape(train_edges)[0])


        # Train-set edge embeddings
        pos_train_edge_embs ,pos_train_sim_matrix,pos_embs_1_train,pos_embs_2_train= get_edge_embeddings(train_edges_sampled,ego_user,DATASET, Flag, flag='pos-train')
        neg_train_edge_embs,neg_train_sim_matrix,neg_embs_1_train,neg_embs_2_train = get_edge_embeddings(train_edges_false, ego_user,DATASET,Flag, flag='neg-train')
        train_edge_embs = pos_train_edge_embs
        train_sim_matrix= pos_train_sim_matrix
        train_embs_1 = pos_embs_1_train
        train_embs_2 = pos_embs_2_train

        # Create train-set edge labels: 1 = real edge, 0 = false edge
        train_edge_labels = np.ones(len(train_edges_sampled))

        # Val-set edge embeddings, labels
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            pos_val_edge_embs,pos_val_sim_matrix,pos_embs_1_val,pos_embs_2_val = get_edge_embeddings(val_edges,ego_user,DATASET,Flag, flag='pos-val')
            neg_val_edge_embs,neg_val_sim_matrix,neg_embs_1_val,neg_embs_2_val = get_edge_embeddings(val_edges_false,ego_user,DATASET,Flag, flag='neg-val')
            val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
            val_edge_labels = np.concatenate((np.ones(len(val_edges)), np.zeros(len(val_edges_false))), axis=0)
            val_sim_matrix = np.concatenate((pos_val_sim_matrix, neg_val_sim_matrix), axis=0)
            val_embs_1 = np.concatenate((pos_embs_1_val, neg_embs_1_val), axis=0)
            val_embs_2 = np.concatenate((pos_embs_2_val, neg_embs_2_val), axis=0)

        # Test-set edge embeddings, labels
        pos_test_edge_embs,pos_test_sim_matrix,pos_embs_1_test,pos_embs_2_test = get_edge_embeddings(test_edges,ego_user,DATASET,Flag, flag='pos-test')
        neg_test_edge_embs ,neg_test_sim_matrix,neg_embs_1_test,neg_embs_2_test= get_edge_embeddings(test_edges_false,ego_user,DATASET,Flag, flag='neg-test')
        test_edge_embs = pos_test_edge_embs
        test_sim_matrix = pos_test_sim_matrix
        test_embs_1 = pos_embs_1_test
        test_embs_2 = pos_embs_2_test

        # Create val-set edge labels: 1 = real edge, 0 = false edge
        test_edge_labels = np.ones(len(test_edges))


    #     # Train logistic regression classifier on train-set edge embeddings
    #     edge_classifier = LogisticRegression(random_state=0)
    #     edge_classifier.fit(train_edge_embs, train_edge_labels)
    #
    #     # Predicted edge scores: probability of being of class "1" (real edge)
    #     if len(val_edges) > 0 and len(val_edges_false) > 0:
    #         val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
    #     test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
    #     # print(test_preds)
    #     #print(np.shape(test_preds))
    #
    #     runtime = time.time() - start_time
    #
    #     # Calculate scores
    #     if len(val_edges) > 0 and len(val_edges_false) > 0:
    #         n2v_val_roc = roc_auc_score(val_edge_labels, val_preds)
    #         # n2v_val_roc_curve = roc_curve(val_edge_labels, val_preds)
    #         n2v_val_ap = average_precision_score(val_edge_labels, val_preds)
    #     else:
    #         n2v_val_roc = None
    #         n2v_val_roc_curve = None
    #         n2v_val_ap = None
    #
    #     n2v_test_roc = roc_auc_score(test_edge_labels, test_preds)
    #     # n2v_test_roc_curve = roc_curve(test_edge_labels, test_preds)
    #     n2v_test_ap = average_precision_score(test_edge_labels, test_preds)
    #
    #
    # # Generate edge scores using simple dot product of node embeddings (like in GAE paper)
    # elif args.edge_score_mode == "dot-product":
    #     score_matrix = np.dot(emb_matrix, emb_matrix.T)
    #     runtime = time.time() - start_time
    #
    #     # Val set scores
    #     if len(val_edges) > 0:
    #         n2v_val_roc, n2v_val_ap = get_roc_score(val_edges, val_edges_false, score_matrix, apply_sigmoid=True)
    #     else:
    #         n2v_val_roc = None
    #         n2v_val_roc_curve = None
    #         n2v_val_ap = None
    #
    #     # Test set scores
    #     n2v_test_roc, n2v_test_ap = get_roc_score(test_edges, test_edges_false, score_matrix, apply_sigmoid=True)
    #
    # else:
    #     print
    #     "Invalid edge_score_mode! Either use edge-emb or dot-product."
    #
    # # Record scores
    # n2v_scores = {}
    #
    # n2v_scores['test_roc'] = n2v_test_roc
    # # n2v_scores['test_roc_curve'] = n2v_test_roc_curve
    # n2v_scores['test_ap'] = n2v_test_ap
    #
    # n2v_scores['val_roc'] = n2v_val_roc
    # # n2v_scores['val_roc_curve'] = n2v_val_roc_curve
    # n2v_scores['val_ap'] = n2v_val_ap
    #
    # n2v_scores['runtime'] = runtime

    return  train_edge_labels,test_edge_labels, train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs,train_embs_1,train_embs_2,test_embs_1,test_embs_2,train_edges_sampled

def deepwalk7(g_train, train_test_split,DATASET,METHOD,res_dir, ego_user, F):
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
  args.input='%s%s-fair-%s-train.txt' % (res_dir, str(ego_user), F)
  args.output = res_dir+METHOD+'-embeds-'+F+'-'+str(ego_user)
  numeric_level = getattr(logging, args.log.upper(), None)
  logging.basicConfig(format=LOGFORMAT)
  logger.setLevel(numeric_level)

  if args.debug:
   sys.excepthook = debug

  adj_train_orig, train_edges, train_edges_false, val_edges, val_edges_false, \
  test_edges, test_edges_false = train_test_split
  g_train=nx.adjacency_matrix(g_train)
  emb_matrix=process(args,g_train,DATASET,METHOD,F,ego_user)

  n2v_scores,train_edge_labels, test_edge_labels, test_preds,train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs,train_embs_1,train_embs_2,test_embs_1,test_embs_2,train_edges_sampled=linkpre_scores7(args, emb_matrix, train_test_split, ego_user,DATASET,METHOD, F)

  return n2v_scores, train_edge_labels,test_edge_labels, test_preds,emb_matrix,train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs,train_embs_1,train_embs_2,test_embs_1,test_embs_2,train_edges_sampled


def linkpre_scores7(args, emb_matrix, train_test_split, ego_user,DATASET,METHOD, Flag):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = train_test_split
    start_time = time.time()
    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
    if args.edge_score_mode == "edge-emb":

        def get_edge_embeddings(edge_list,ego_user,DATASET, Flag, flag):
            embs = []
            sim_matrix=[]
            embs_1=[]
            embs_2 = []
            for edge in edge_list:
                node1 = edge[0]
                node2 = edge[1]
                emb1 = emb_matrix[node1]
                #print(np.shape(emb1))
                emb2 = emb_matrix[node2]
                edge_emb = np.multiply(emb1, emb2)
                sim = np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
                #edge_emb = np.array(emb1) + np.array(emb2)
                print(np.shape(edge_emb))
                embs.append(edge_emb)
                embs_1.append(emb1)
                embs_2.append(emb2)
                sim_matrix.append(sim)
            embs = np.array(embs)
            sim_matrix = np.array(sim_matrix)
            embs_1=np.array(embs_1)
            embs_2 =np.array(embs_2)

            # #with open('/Users/xiulingwang/Downloads/line-master/data/embds/' + str(ego_user) + flag + '-' + Flag, 'w') as f:
            # with open('E:/python/banlance/code/' + DATASET + '/' + METHOD + '/embeds/' + Flag + '-' + str(ego_user) + flag,'w') as f:
            # #with open('/Users/xiulingwang/Downloads/'+DATASET+'/line/3-split/' + str(ego_user)+ '-' + flag + '-' + Flag+'-' +'embds','w') as f:
            #     f.write('%d %d\n' % (edge_list.shape[0], args.representation_size))
            #     for i in range(edge_list.shape[0]):
            #         e = ' '.join(map(lambda x: str(x), embs[i]))
            #         f.write('%s %s %s\n' % (str(edge_list[i][0]), str(edge_list[i][1]), e))

            return embs,sim_matrix,embs_1,embs_2

        edgeall = list([list(edge_tuple) for edge_tuple in train_edges])

        train_edges = random.sample(edgeall, np.shape(test_edges)[0])
        print(np.shape(train_edges))
        print(np.shape(test_edges))
        # exit()


        # Train-set edge embeddings
        pos_train_edge_embs ,pos_train_sim_matrix,pos_embs_1_train,pos_embs_2_train= get_edge_embeddings(train_edges,ego_user,DATASET, Flag, flag='pos-train')
        neg_train_edge_embs,neg_train_sim_matrix,neg_embs_1_train,neg_embs_2_train = get_edge_embeddings(train_edges_false, ego_user,DATASET,Flag, flag='neg-train')
        train_edge_embs = np.concatenate((pos_train_edge_embs, neg_train_edge_embs), axis=0)
        train_sim_matrix= np.concatenate((pos_train_sim_matrix, neg_train_sim_matrix), axis=0)
        train_embs_1 = np.concatenate((pos_embs_1_train, neg_embs_1_train), axis=0)
        train_embs_2 = np.concatenate((pos_embs_2_train, neg_embs_2_train), axis=0)

        # Create train-set edge labels: 1 = real edge, 0 = false edge
        train_edge_labels = np.concatenate((np.ones(len(train_edges)), np.zeros(len(train_edges_false))), axis=0)

        # Val-set edge embeddings, labels
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            pos_val_edge_embs,pos_val_sim_matrix,pos_embs_1_val,pos_embs_2_val = get_edge_embeddings(val_edges,ego_user,DATASET,Flag, flag='pos-val')
            neg_val_edge_embs,neg_val_sim_matrix,neg_embs_1_val,neg_embs_2_val = get_edge_embeddings(val_edges_false,ego_user,DATASET,Flag, flag='neg-val')
            val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
            val_edge_labels = np.concatenate((np.ones(len(val_edges)), np.zeros(len(val_edges_false))), axis=0)
            val_sim_matrix = np.concatenate((pos_val_sim_matrix, neg_val_sim_matrix), axis=0)
            val_embs_1 = np.concatenate((pos_embs_1_val, neg_embs_1_val), axis=0)
            val_embs_2 = np.concatenate((pos_embs_2_val, neg_embs_2_val), axis=0)

        # Test-set edge embeddings, labels
        pos_test_edge_embs,pos_test_sim_matrix,pos_embs_1_test,pos_embs_2_test = get_edge_embeddings(test_edges,ego_user,DATASET,Flag, flag='pos-test')
        neg_test_edge_embs ,neg_test_sim_matrix,neg_embs_1_test,neg_embs_2_test= get_edge_embeddings(test_edges_false,ego_user,DATASET,Flag, flag='neg-test')
        test_edge_embs = np.concatenate((pos_test_edge_embs, neg_test_edge_embs), axis=0)
        test_sim_matrix = np.concatenate((pos_test_sim_matrix, neg_test_sim_matrix), axis=0)
        test_embs_1 = np.concatenate((pos_embs_1_test, neg_embs_1_test), axis=0)
        test_embs_2 = np.concatenate((pos_embs_2_test, neg_embs_2_test), axis=0)

        # Create val-set edge labels: 1 = real edge, 0 = false edge
        test_edge_labels = np.concatenate((np.ones(len(test_edges)), np.zeros(len(test_edges_false))), axis=0)

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

    return n2v_scores, train_edge_labels,test_edge_labels, test_preds,train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs,train_embs_1,train_embs_2,test_embs_1,test_embs_2,train_edges


def process_pytorch(args,adj_train,DATASET,METHOD,F,ego_user,res_dir,dp,sigma):

  global begin

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

  print("Walking...")

  if dp == 6:
    walks = graph.build_defense_walks(sigma,adj_train,G, num_paths=args.number_walks,
                                          path_length=args.walk_length, alpha=0, rand=random.Random(args.seed))
  else:
    walks = graph.build_deepwalk_corpus(G, num_paths=args.number_walks,
                                      path_length=args.walk_length, alpha=0, rand=random.Random(args.seed))

  write_walks2(args, walks, DATASET, METHOD, F, ego_user,res_dir)

  print("Training...")

  # print(walks)

  if dp==1:
    model = word2vec.ModWord2Vec_dp(walks, size=args.representation_size, window=args.window_size, min_count=0, sg=1, hs=1, workers=args.workers,compute_loss=True,budget=sigma)
  elif dp==0 or dp==6:
    model = word2vec.ModWordVec(walks, size=args.representation_size, window=args.window_size, min_count=0, sg=1,
                                      hs=1, workers=args.workers,compute_loss=True)



  print('%%%%%%%%%%%%%')
  # args.output='E:\\python\\banlance\\code\\' + DATASET + '\\' + METHOD + '-embeds-' + F + '-' + str(ego_user)
  emb_matrix=model.save_emb(args.output, len(G.nodes()))

  #
  # # Store embeddings mapping
  # emb_mappings = model.wv
  #
  # # Create node embeddings matrix (rows = nodes, columns = embedding features)
  # emb_list = []
  # for node_index in range(0, adj_train.shape[0]):
  #     node_str = str(node_index)
  #     node_emb = emb_mappings[node_str]
  #     emb_list.append(node_emb)
  # emb_matrix = np.vstack(emb_list)

  with open('E:\\python\\banlance\\code\\' + DATASET + '\\' + METHOD + '-embeds-' + F + '-' + str(ego_user), 'w') as f:
      f.write('%d %d\n' % (adj_train.shape[0], args.representation_size))
      for i in range(adj_train.shape[0]):
          e = ' '.join(map(lambda x: str(x), emb_matrix[i]))
          f.write('%s %s\n' % (str(i), e))

  # exit()
  return emb_matrix

def process_pytorch1(args,adj_train,DATASET,METHOD,F,ego_user,res_dir,dp,sigma,train_edges,test_edges):

  global begin

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

  print("Walking...")
  walks = graph.build_deepwalk_corpus(G, num_paths=args.number_walks,
                                      path_length=args.walk_length, alpha=0, rand=random.Random(args.seed))

  write_walks2(args, walks, DATASET, METHOD, F, ego_user,res_dir)

  print("Training...")

  print(walks)

  model = word2vec.ModWord2Vec_defense(train_edges,test_edges,len(G.nodes()),args.output, walks,size=args.representation_size, window=args.window_size, min_count=0, sg=1,
                                   hs=1, workers=args.workers,compute_loss=True)
  print('%%%%%%%%%%%%%')
  # args.output='E:\\python\\banlance\\code\\' + DATASET + '\\' + METHOD + '-embeds-' + F + '-' + str(ego_user)
  emb_matrix=model.save_emb(args.output, len(G.nodes()))

  #
  # # Store embeddings mapping
  # emb_mappings = model.wv
  #
  # # Create node embeddings matrix (rows = nodes, columns = embedding features)
  # emb_list = []
  # for node_index in range(0, adj_train.shape[0]):
  #     node_str = str(node_index)
  #     node_emb = emb_mappings[node_str]
  #     emb_list.append(node_emb)
  # emb_matrix = np.vstack(emb_list)

  with open('E:\\python\\banlance\\code\\' + DATASET + '\\' + METHOD + '-embeds-' + F + '-' + str(ego_user), 'w') as f:
      f.write('%d %d\n' % (adj_train.shape[0], args.representation_size))
      for i in range(adj_train.shape[0]):
          e = ' '.join(map(lambda x: str(x), emb_matrix[i]))
          f.write('%s %s\n' % (str(i), e))

  # exit()
  return emb_matrix


def process_pytorch2(args,adj_train,DATASET,METHOD,F,ego_user,res_dir,dp,sigma,train_edges,test_edges):

  global begin

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

  print("Walking...")
  walks = graph.build_deepwalk_corpus(G, num_paths=args.number_walks,
                                      path_length=args.walk_length, alpha=0, rand=random.Random(args.seed))

  write_walks2(args, walks, DATASET, METHOD, F, ego_user,res_dir)

  print("Training...")

  print(walks)

  model = word2vec.ModWord2Vec_defense2(train_edges,test_edges,len(G.nodes()),args.output, walks,size=args.representation_size, window=args.window_size, min_count=0, sg=1,
                                   hs=1, workers=args.workers,compute_loss=True)
  print('%%%%%%%%%%%%%')
  # args.output='E:\\python\\banlance\\code\\' + DATASET + '\\' + METHOD + '-embeds-' + F + '-' + str(ego_user)
  emb_matrix=model.save_emb(args.output, len(G.nodes()))

  #
  # # Store embeddings mapping
  # emb_mappings = model.wv
  #
  # # Create node embeddings matrix (rows = nodes, columns = embedding features)
  # emb_list = []
  # for node_index in range(0, adj_train.shape[0]):
  #     node_str = str(node_index)
  #     node_emb = emb_mappings[node_str]
  #     emb_list.append(node_emb)
  # emb_matrix = np.vstack(emb_list)

  with open('E:\\python\\banlance\\code\\' + DATASET + '\\' + METHOD + '-embeds-' + F + '-' + str(ego_user), 'w') as f:
      f.write('%d %d\n' % (adj_train.shape[0], args.representation_size))
      for i in range(adj_train.shape[0]):
          e = ' '.join(map(lambda x: str(x), emb_matrix[i]))
          f.write('%s %s\n' % (str(i), e))

  # exit()
  return emb_matrix


def process_pytorch3(args,adj_train,DATASET,METHOD,F,ego_user,res_dir,dp,sigma,train_edges,test_edges):

  global begin

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

  print("Walking...")
  walks = graph.build_deepwalk_corpus(G, num_paths=args.number_walks,
                                      path_length=args.walk_length, alpha=0, rand=random.Random(args.seed))

  write_walks2(args, walks, DATASET, METHOD, F, ego_user,res_dir)

  print("Training...")

  print(walks)

  model = word2vec.ModWord2Vec_defense3(F, res_dir,train_edges,test_edges,len(G.nodes()),args.output, walks,size=args.representation_size, window=args.window_size, min_count=0, sg=1,
                                   hs=1, workers=args.workers,compute_loss=True)
  print('%%%%%%%%%%%%%')
  # args.output='E:\\python\\banlance\\code\\' + DATASET + '\\' + METHOD + '-embeds-' + F + '-' + str(ego_user)
  emb_matrix=model.save_emb(args.output, len(G.nodes()))

  #
  # # Store embeddings mapping
  # emb_mappings = model.wv
  #
  # # Create node embeddings matrix (rows = nodes, columns = embedding features)
  # emb_list = []
  # for node_index in range(0, adj_train.shape[0]):
  #     node_str = str(node_index)
  #     node_emb = emb_mappings[node_str]
  #     emb_list.append(node_emb)
  # emb_matrix = np.vstack(emb_list)

  # exit()
  return emb_matrix

def process_pytorch5(args,adj_train,DATASET,METHOD,F,ego_user,res_dir,dp,sigma):

  global begin

  if args.format == "adjlist":
    G = graph.load_adjacencylist(args.input, undirected=args.undirected)
  elif args.format == "edgelist":
    G = graph.load_edgelist(args.input, undirected=args.undirected)
  elif args.format == "mat":
    G = graph.load_matfile(args.input, variable_name=args.matfile_variable_name, undirected=args.undirected)
  else:
    raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', 'mat'" % args.format)

  # print(np.shape(adj_train)[0])

  # for i in range(np.shape(adj_train)[0]):
  #     if i not in G.keys():
  #         G[i].append(i)



  print("Number of nodes: {}".format(len(G.nodes())))

  num_walks = len(G.nodes()) * args.number_walks

  print("Number of walks: {}".format(num_walks))

  data_size = num_walks * args.walk_length

  print("Data size (walks*length): {}".format(data_size))

  print("Walking...")
  walks = graph.build_deepwalk_corpus(G, num_paths=args.number_walks,
                                      path_length=args.walk_length, alpha=0, rand=random.Random(args.seed))

  write_walks2(args, walks, DATASET, METHOD, F, ego_user,res_dir)

  print("Training...")

  print(walks)

  if dp==1:
    model = word2vec.ModWord2Vec_dp(walks, size=args.representation_size, window=args.window_size, min_count=0, sg=1, hs=1, workers=args.workers,compute_loss=True,budget=sigma)
  elif dp==0 or dp==5:
    model = word2vec.ModWord2Vec(walks, size=args.representation_size, window=args.window_size, min_count=0, sg=1,
                                      hs=1, workers=args.workers,compute_loss=True)

  print('%%%%%%%%%%%%%')
  # args.output='E:\\python\\banlance\\code\\' + DATASET + '\\' + METHOD + '-embeds-' + F + '-' + str(ego_user)
  emb_matrix=model.save_emb(args.output, len(G.nodes()))

  #
  # # Store embeddings mapping
  # emb_mappings = model.wv
  #
  # # Create node embeddings matrix (rows = nodes, columns = embedding features)
  # emb_list = []
  # for node_index in range(0, adj_train.shape[0]):
  #     node_str = str(node_index)
  #     node_emb = emb_mappings[node_str]
  #     emb_list.append(node_emb)
  # emb_matrix = np.vstack(emb_list)

  with open('E:\\python\\banlance\\code\\' + DATASET + '\\' + METHOD + '-embeds-' + F + '-' + str(ego_user), 'w') as f:
      f.write('%d %d\n' % (adj_train.shape[0], args.representation_size))
      for i in range(adj_train.shape[0]):
          e = ' '.join(map(lambda x: str(x), emb_matrix[i]))
          f.write('%s %s\n' % (str(i), e))

  # exit()
  return emb_matrix

def process_pytorch9(args,adj_train,DATASET,METHOD,F,ego_user,res_dir,dp,sigma):

  global begin

  if args.format == "adjlist":
    G = graph.load_adjacencylist(args.input, undirected=args.undirected)
  elif args.format == "edgelist":
    G = graph.load_edgelist(args.input, undirected=args.undirected)
  elif args.format == "mat":
    G = graph.load_matfile(args.input, variable_name=args.matfile_variable_name, undirected=args.undirected)
  else:
    raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', 'mat'" % args.format)

  # print(np.shape(adj_train)[0])

  # for i in range(np.shape(adj_train)[0]):
  #     if i not in G.keys():
  #         G[i].append(i)



  print("Number of nodes: {}".format(len(G.nodes())))

  num_walks = len(G.nodes()) * args.number_walks

  print("Number of walks: {}".format(num_walks))

  data_size = num_walks * args.walk_length

  print("Data size (walks*length): {}".format(data_size))

  print("Walking...")
  walks = graph.build_deepwalk_corpus(G, num_paths=args.number_walks,
                                      path_length=args.walk_length, alpha=0, rand=random.Random(args.seed))

  write_walks2(args, walks, DATASET, METHOD, F, ego_user,res_dir)

  print("Training...")

  print(walks)

  if dp==1:
    model = word2vec.ModWord2Vec_dp(walks, size=args.representation_size, window=args.window_size, min_count=0, sg=1, hs=1, workers=args.workers,compute_loss=True,budget=sigma)
  elif dp==0 or dp==5:
    model = word2vec.ModWord2Vec(walks, size=args.representation_size, window=args.window_size, min_count=0, sg=1,
                                      hs=1, workers=args.workers,compute_loss=True)

  elif dp==9:
    model = word2vec.ModWord2Vec_9(walks, size=args.representation_size, window=args.window_size, min_count=0, sg=1,
                                   hs=1, workers=args.workers, compute_loss=True)

  print('%%%%%%%%%%%%%')
  # args.output='E:\\python\\banlance\\code\\' + DATASET + '\\' + METHOD + '-embeds-' + F + '-' + str(ego_user)
  emb_matrix=model.save_emb(args.output, len(G.nodes()))

  #
  # # Store embeddings mapping
  # emb_mappings = model.wv
  #
  # # Create node embeddings matrix (rows = nodes, columns = embedding features)
  # emb_list = []
  # for node_index in range(0, adj_train.shape[0]):
  #     node_str = str(node_index)
  #     node_emb = emb_mappings[node_str]
  #     emb_list.append(node_emb)
  # emb_matrix = np.vstack(emb_list)

  with open('E:\\python\\banlance\\code\\' + DATASET + '\\' + METHOD + '-embeds-' + F + '-' + str(ego_user), 'w') as f:
      f.write('%d %d\n' % (adj_train.shape[0], args.representation_size))
      for i in range(adj_train.shape[0]):
          e = ' '.join(map(lambda x: str(x), emb_matrix[i]))
          f.write('%s %s\n' % (str(i), e))

  # exit()
  return emb_matrix



def deepwalk8(g_train, train_test_split,DATASET,METHOD,res_dir, ego_user, F,dp,sigma):
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
  args.input='%s%s-fair-%s-train.txt' % (res_dir, str(ego_user), F)
  args.output = res_dir+METHOD+'-embeds-'+F+'-'+str(ego_user)
  numeric_level = getattr(logging, args.log.upper(), None)
  logging.basicConfig(format=LOGFORMAT)
  logger.setLevel(numeric_level)

  if args.debug:
   sys.excepthook = debug

  adj_train_orig, train_edges, train_edges_false, val_edges, val_edges_false, \
  test_edges, test_edges_false = train_test_split
  g_train=nx.adjacency_matrix(g_train)
  if dp==0 or dp==1 or dp==6:
    emb_matrix=process_pytorch(args,g_train,DATASET,METHOD,F,ego_user,res_dir,dp,sigma)

  if dp == 2:
    emb_matrix = process_pytorch1(args, g_train, DATASET, METHOD, F, ego_user, res_dir, dp, sigma,train_edges,test_edges)

  if dp == 3:
    emb_matrix = process_pytorch2(args, g_train, DATASET, METHOD, F, ego_user, res_dir, dp, sigma,train_edges,test_edges)

  if dp == 4:
    emb_matrix = process_pytorch3(args, g_train, DATASET, METHOD, F, ego_user, res_dir, dp, sigma,train_edges,test_edges)

  if dp==5:
    emb_matrix=process_pytorch5(args,g_train,DATASET,METHOD,F,ego_user,res_dir,dp,sigma)

  # if dp==9:
  #   emb_matrix=process_pytorch9(args,g_train,DATASET,METHOD,F,ego_user,res_dir,dp,sigma)

  train_edge_labels, test_edge_labels, train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs,train_embs_1,train_embs_2,test_embs_1,test_embs_2,train_edges_sampled=linkpre_scores8(args, emb_matrix, train_test_split, ego_user,DATASET,METHOD, F)

  return train_edge_labels,test_edge_labels, emb_matrix,train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs,train_embs_1,train_embs_2,test_embs_1,test_embs_2,train_edges_sampled


def linkpre_scores8(args, emb_matrix, train_test_split, ego_user,DATASET,METHOD, Flag):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = train_test_split
    start_time = time.time()
    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
    if args.edge_score_mode == "edge-emb":

        def get_edge_embeddings(edge_list,ego_user,DATASET, Flag, flag):
            embs = []
            sim_matrix=[]
            embs_1=[]
            embs_2 = []
            tsts=[]
            for edge in edge_list:
                node1 = edge[0]
                node2 = edge[1]
                emb1 = emb_matrix[node1]
                # print(np.shape(emb1))
                # print(type(emb1))
                emb2 = emb_matrix[node2]
                edge_emb = np.multiply(emb1, emb2)
                sim = np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
                #edge_emb = np.array(emb1) + np.array(emb2)
                # print(np.shape(edge_emb))
                sim2 = np.dot(emb1, emb2)
                # sim3 = np.sqrt(np.sum(np.sqrt(np.array(emb1)-np.array(emb2))))
                # print(sim3)
                sim3=np.linalg.norm(np.array(emb1)-np.array(emb2))
                # print(sim3)
                sim4=1/(1+sim3)
                embs.append(edge_emb)
                embs_1.append(emb1)
                embs_2.append(emb2)
                sim_matrix.append([sim,sim2,sim3,sim4])

                tst = [node1, node2, sim,sim2,sim3,sim4]
                tsts.append(tst)

            embs = np.array(embs)
            sim_matrix = np.array(sim_matrix)
            embs_1=np.array(embs_1)
            embs_2 =np.array(embs_2)

            name = ['node1', 'node2', 'sim1', 'sim2', 'sim3','sim4']
            result = pd.DataFrame(columns=name, data=tsts)
            result.to_csv("{}{}-similarity.csv".format(Flag, flag))

            # #with open('/Users/xiulingwang/Downloads/line-master/data/embds/' + str(ego_user) + flag + '-' + Flag, 'w') as f:
            # with open('E:/python/banlance/code/' + DATASET + '/' + METHOD + '/embeds/' + Flag + '-' + str(ego_user) + flag,'w') as f:
            # #with open('/Users/xiulingwang/Downloads/'+DATASET+'/line/3-split/' + str(ego_user)+ '-' + flag + '-' + Flag+'-' +'embds','w') as f:
            #     f.write('%d %d\n' % (edge_list.shape[0], args.representation_size))
            #     for i in range(edge_list.shape[0]):
            #         e = ' '.join(map(lambda x: str(x), embs[i]))
            #         f.write('%s %s %s\n' % (str(edge_list[i][0]), str(edge_list[i][1]), e))

            return embs,sim_matrix,embs_1,embs_2

        edgeall = list([list(edge_tuple) for edge_tuple in train_edges])

        # train_edges_sampled = random.sample(edgeall, np.shape(test_edges)[0])
        train_edges_sampled = random.sample(edgeall, np.shape(test_edges)[0])


        # Train-set edge embeddings
        pos_train_edge_embs0 ,pos_train_sim_matrix0,pos_embs_1_train0,pos_embs_2_train0= get_edge_embeddings(edgeall ,ego_user,DATASET, Flag, flag='pos-train-all')

        pos_train_edge_embs ,pos_train_sim_matrix,pos_embs_1_train,pos_embs_2_train= get_edge_embeddings(train_edges_sampled,ego_user,DATASET, Flag, flag='pos-train')
        neg_train_edge_embs,neg_train_sim_matrix,neg_embs_1_train,neg_embs_2_train = get_edge_embeddings(train_edges_false, ego_user,DATASET,Flag, flag='neg-train')
        train_edge_embs = pos_train_edge_embs
        train_sim_matrix= pos_train_sim_matrix
        train_embs_1 = pos_embs_1_train
        train_embs_2 = pos_embs_2_train

        # Create train-set edge labels: 1 = real edge, 0 = false edge
        train_edge_labels = np.ones(len(train_edges_sampled))

        # Val-set edge embeddings, labels
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            pos_val_edge_embs,pos_val_sim_matrix,pos_embs_1_val,pos_embs_2_val = get_edge_embeddings(val_edges,ego_user,DATASET,Flag, flag='pos-val')
            neg_val_edge_embs,neg_val_sim_matrix,neg_embs_1_val,neg_embs_2_val = get_edge_embeddings(val_edges_false,ego_user,DATASET,Flag, flag='neg-val')
            val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
            val_edge_labels = np.concatenate((np.ones(len(val_edges)), np.zeros(len(val_edges_false))), axis=0)
            val_sim_matrix = np.concatenate((pos_val_sim_matrix, neg_val_sim_matrix), axis=0)
            val_embs_1 = np.concatenate((pos_embs_1_val, neg_embs_1_val), axis=0)
            val_embs_2 = np.concatenate((pos_embs_2_val, neg_embs_2_val), axis=0)

        # Test-set edge embeddings, labels
        pos_test_edge_embs,pos_test_sim_matrix,pos_embs_1_test,pos_embs_2_test = get_edge_embeddings(test_edges,ego_user,DATASET,Flag, flag='pos-test')
        neg_test_edge_embs ,neg_test_sim_matrix,neg_embs_1_test,neg_embs_2_test= get_edge_embeddings(test_edges_false,ego_user,DATASET,Flag, flag='neg-test')
        test_edge_embs = pos_test_edge_embs
        test_sim_matrix = pos_test_sim_matrix
        test_embs_1 = pos_embs_1_test
        test_embs_2 = pos_embs_2_test

        # Create val-set edge labels: 1 = real edge, 0 = false edge
        test_edge_labels = np.ones(len(test_edges))


    #     # Train logistic regression classifier on train-set edge embeddings
    #     edge_classifier = LogisticRegression(random_state=0)
    #     edge_classifier.fit(train_edge_embs, train_edge_labels)
    #
    #     # Predicted edge scores: probability of being of class "1" (real edge)
    #     if len(val_edges) > 0 and len(val_edges_false) > 0:
    #         val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
    #     test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
    #     print(test_preds)
    #     print(np.shape(test_preds))
    #
    #     runtime = time.time() - start_time
    #
    #     # Calculate scores
    #     if len(val_edges) > 0 and len(val_edges_false) > 0:
    #         n2v_val_roc = roc_auc_score(val_edge_labels, val_preds)
    #         # n2v_val_roc_curve = roc_curve(val_edge_labels, val_preds)
    #         n2v_val_ap = average_precision_score(val_edge_labels, val_preds)
    #     else:
    #         n2v_val_roc = None
    #         n2v_val_roc_curve = None
    #         n2v_val_ap = None
    #
    #     n2v_test_roc = roc_auc_score(test_edge_labels, test_preds)
    #     # n2v_test_roc_curve = roc_curve(test_edge_labels, test_preds)
    #     n2v_test_ap = average_precision_score(test_edge_labels, test_preds)
    #
    #
    # # Generate edge scores using simple dot product of node embeddings (like in GAE paper)
    # elif args.edge_score_mode == "dot-product":
    #     score_matrix = np.dot(emb_matrix, emb_matrix.T)
    #     runtime = time.time() - start_time
    #
    #     # Val set scores
    #     if len(val_edges) > 0:
    #         n2v_val_roc, n2v_val_ap = get_roc_score(val_edges, val_edges_false, score_matrix, apply_sigmoid=True)
    #     else:
    #         n2v_val_roc = None
    #         n2v_val_roc_curve = None
    #         n2v_val_ap = None
    #
    #     # Test set scores
    #     n2v_test_roc, n2v_test_ap = get_roc_score(test_edges, test_edges_false, score_matrix, apply_sigmoid=True)
    #
    # else:
    #     print
    #     "Invalid edge_score_mode! Either use edge-emb or dot-product."
    #
    # # Record scores
    n2v_scores = {}
    #
    # n2v_scores['test_roc'] = n2v_test_roc
    # # n2v_scores['test_roc_curve'] = n2v_test_roc_curve
    # n2v_scores['test_ap'] = n2v_test_ap
    #
    # n2v_scores['val_roc'] = n2v_val_roc
    # # n2v_scores['val_roc_curve'] = n2v_val_roc_curve
    # n2v_scores['val_ap'] = n2v_val_ap
    #
    # n2v_scores['runtime'] = runtime

    return train_edge_labels,test_edge_labels,train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs,train_embs_1,train_embs_2,test_embs_1,test_embs_2,train_edges_sampled

def discriminator(train_edges,test_edges,emb_matrix):

    start_time = time.time()
    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2

    def get_edge_embeddings(edge_list):
        embs = []
        sim_matrix=[]
        embs_1=[]
        embs_2 = []
        for edge in edge_list:
            node1 = edge[0]
            node2 = edge[1]
            emb1 = emb_matrix[node1]
            #print(np.shape(emb1))
            emb2 = emb_matrix[node2]
            edge_emb = np.multiply(emb1, emb2)
            sim = np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
            #edge_emb = np.array(emb1) + np.array(emb2)
            # print(np.shape(edge_emb))
            embs.append(edge_emb)
            embs_1.append(emb1)
            embs_2.append(emb2)
            sim_matrix.append(sim)
        embs = np.array(embs)
        sim_matrix = np.array(sim_matrix)
        embs_1=np.array(embs_1)
        embs_2 =np.array(embs_2)

        # #with open('/Users/xiulingwang/Downloads/line-master/data/embds/' + str(ego_user) + flag + '-' + Flag, 'w') as f:
        # with open('E:/python/banlance/code/' + DATASET + '/' + METHOD + '/embeds/' + Flag + '-' + str(ego_user) + flag,'w') as f:
        # #with open('/Users/xiulingwang/Downloads/'+DATASET+'/line/3-split/' + str(ego_user)+ '-' + flag + '-' + Flag+'-' +'embds','w') as f:
        #     f.write('%d %d\n' % (edge_list.shape[0], args.representation_size))
        #     for i in range(edge_list.shape[0]):
        #         e = ' '.join(map(lambda x: str(x), embs[i]))
        #         f.write('%s %s %s\n' % (str(edge_list[i][0]), str(edge_list[i][1]), e))

        return embs,sim_matrix,embs_1,embs_2

    edgeall = list([list(edge_tuple) for edge_tuple in train_edges])

    # train_edges_sampled = random.sample(edgeall, np.shape(test_edges)[0])
    train_edges_sampled = random.sample(edgeall, np.shape(test_edges)[0])


    # Train-set edge embeddings
    pos_train_edge_embs ,pos_train_sim_matrix,pos_embs_1_train,pos_embs_2_train= get_edge_embeddings(train_edges_sampled)
    # neg_train_edge_embs,neg_train_sim_matrix,neg_embs_1_train,neg_embs_2_train = get_edge_embeddings(train_edges_false, ego_user,DATASET,Flag, flag='neg-train')
    train_edge_embs = pos_train_edge_embs
    train_sim_matrix= pos_train_sim_matrix
    train_embs_1 = pos_embs_1_train
    train_embs_2 = pos_embs_2_train

    # Create train-set edge labels: 1 = real edge, 0 = false edge
    train_edge_labels = np.ones(len(train_edges_sampled))


    # Test-set edge embeddings, labels
    pos_test_edge_embs,pos_test_sim_matrix,pos_embs_1_test,pos_embs_2_test = get_edge_embeddings(test_edges)
    # neg_test_edge_embs ,neg_test_sim_matrix,neg_embs_1_test,neg_embs_2_test= get_edge_embeddings(test_edges_false,ego_user,DATASET,Flag, flag='neg-test')
    test_edge_embs = pos_test_edge_embs
    test_sim_matrix = pos_test_sim_matrix
    test_embs_1 = pos_embs_1_test
    test_embs_2 = pos_embs_2_test

    # Create val-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.ones(len(test_edges))


    ###########sim_svm

    train_edges_list = np.array(train_edges_sampled)
    print(train_edges_list)
    test_edges_list = test_edges
    print(test_edges_list)

    edges_list = np.concatenate((train_edges_list, test_edges_list), axis=0)

    print(type(train_edges_list))
    print(type(test_edges_list))
    print(type(edges_list))

    print(np.shape(train_edges_list))
    print(np.shape(test_edges_list))
    print(np.shape(edges_list))

    ylabel = [1] * train_sim_matrix.shape[0] + [0] * test_sim_matrix.shape[0]

    # print(train_sim_matrix)
    # print(test_sim_matrix)

    sim_matrix = np.concatenate((train_sim_matrix, test_sim_matrix), axis=0)
    # print(sim_matrix)
    print(np.shape(train_sim_matrix))
    print(np.shape(test_sim_matrix))
    sim_matrix = sim_matrix.reshape(-1, 1)
    # print(sim_matrix)
    print(np.shape(sim_matrix))
    # exit()

    sim_matrix_train = train_sim_matrix
    sim_matrix_test = test_sim_matrix

    sim_matrix_train = sim_matrix_train.reshape(-1, 1)
    sim_matrix_test = sim_matrix_test.reshape(-1, 1)

    print(np.shape(sim_matrix_train))
    print(np.shape(sim_matrix_test))

    from sklearn.model_selection import train_test_split

    ylabel1 = ylabel
    ylable1 = np.reshape(len(ylabel1), 1)

    # print((edges_list))
    # print((ylabel1))
    # print(np.shape(ylabel1))
    # print(np.shape(edges_list))
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
                                                                                test_size=0.1, random_state=42)

    X_test_train, X_test_test, y_test_train, y_test_test = train_test_split(sim_matrix_test, y_label_test,
                                                                            test_size=0.1, random_state=42)

    X_train = np.concatenate((X_train_train, X_test_train), axis=0)
    X_test = np.concatenate((X_train_test, X_test_test), axis=0)
    y_train = np.concatenate((y_train_train, y_test_train), axis=0)
    y_test = np.concatenate((y_train_test, y_test_test), axis=0)

    from sklearn import metrics
    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16, 18), random_state=1,
                        max_iter=500)

    mlp.fit(X_train, y_train[:, 2])

    loss=mlp.loss_

    # print("Training set score: %f" % mlp.score(X_train, y_train[:, 2]))
    # print("Test set score: %f" % mlp.score(X_test, y_test[:, 2]))
    #
    # y_score = mlp.predict(X_test)
    # print(metrics.f1_score(y_test[:, 2], y_score, average='micro'))
    # print(metrics.classification_report(y_test[:, 2], y_score, labels=range(3)))

    return loss

