#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Graph utilities."""

import logging
import sys
from io import open
from os import path
from time import time
from glob import glob
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
import random
from random import shuffle
from itertools import product,permutations
from scipy.io import loadmat
from scipy.sparse import issparse


logger = logging.getLogger("deepwalk")


__author__ = "Bryan Perozzi"
__email__ = "bperozzi@cs.stonybrook.edu"

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

class Graph(defaultdict):
  """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""  
  def __init__(self):
    super(Graph, self).__init__(list)

  def nodes(self):
    return self.keys()

  def adjacency_iter(self):
    return self.iteritems()

  def subgraph(self, nodes={}):
    subgraph = Graph()
    
    for n in nodes:
      if n in self:
        subgraph[n] = [x for x in self[n] if x in nodes]
        
    return subgraph

  def make_undirected(self):
  
    t0 = time()

    for v in list(self):
      for other in self[v]:
        if v != other:
          self[other].append(v)
    
    t1 = time()
    logger.info('make_directed: added missing edges {}s'.format(t1-t0))

    self.make_consistent()
    return self

  def make_consistent(self):
    t0 = time()
    for k in iterkeys(self):
      self[k] = list(sorted(set(self[k])))
    
    t1 = time()
    logger.info('make_consistent: made consistent in {}s'.format(t1-t0))

    self.remove_self_loops()

    return self

  def remove_self_loops(self):

    removed = 0
    t0 = time()

    for x in self:
      if x in self[x]: 
        self[x].remove(x)
        removed += 1
    
    t1 = time()

    logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1-t0)))
    return self

  def check_self_loops(self):
    for x in self:
      for y in self[x]:
        if x == y:
          return True
    
    return False

  def has_edge(self, v1, v2):
    if v2 in self[v1] or v1 in self[v2]:
      return True
    return False

  def degree(self, nodes=None):
    if isinstance(nodes, Iterable):
      return {v:len(self[v]) for v in nodes}
    else:
      return len(self[nodes])

  def order(self):
    "Returns the number of nodes in the graph"
    return len(self)    

  def number_of_edges(self):
    "Returns the number of nodes in the graph"
    return sum([self.degree(x) for x in self.keys()])/2

  def number_of_nodes(self):
    "Returns the number of nodes in the graph"
    return self.order()


  def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
    """ Returns a truncated random walk.

        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    G = self
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(list(G.keys()))]

    while len(path) < path_length:
      cur = path[-1]
      if len(G[cur]) > 0:
        if rand.random() >= alpha:
          path.append(rand.choice(G[cur]))
        else:
          path.append(path[0])
      else:
        break
    return [str(node) for node in path]


  def random_walk1(self, path_length, alpha=0, rand=random.Random(), start=None):
    """ Returns a truncated random walk.

        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    G = self
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(list(G.keys()))]

    while len(path) < path_length:
      cur = path[-1]
      if len(G[cur]) > 0:
        if rand.random() >= alpha:
          path.append(rand.choice(G[cur]))
        else:
          path.append(path[0])
      else:
        break
    return [int(node) for node in path]

# TODO add build_walks in here

def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                      rand=random.Random(0)):
  walks = []

  nodes = list(G.nodes())
  
  for cnt in range(num_paths):
    rand.shuffle(nodes)
    for node in nodes:
      walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))
  
  return walks

import networkx as nx
import numpy as np

def add_laplace_noise(data_list, μ=0, b=2):
    laplace_noise = np.random.laplace(μ, b, np.shape(data_list))
    return laplace_noise + data_list

def build_defense_walks(b,adj_train,G, num_paths, path_length, alpha=0,
                          rand=random.Random(0)):
  walks = []

  nodes = list(G.nodes())
  print(nodes)


  u=0

  for cnt in range(num_paths):
    # adj = nx.adjacency_matrix(G)
    adj = np.array(adj_train.todense())
    adj1 = add_laplace_noise(np.array(adj), u, b)
    rand.shuffle(nodes)

    # G1=nx.graph(adj1)

    for node in nodes:
      # walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))

      walk = [node]

      while len(walk) < path_length:

        cur = walk[-1]
        cur_nbrs = adj1[cur,:]
        cur_nbrs =np.argsort(cur_nbrs)[::-1]

        nbr_choice=list(cur_nbrs[0:int(0.5*len(nodes))])


        next = random.choice(nbr_choice)
        walk.append(next)
      walks.append(walk)
      # print(walks)
      if walks[0]==0:
        print('&&&')

  return walks


def fair_walks(G, num_walks, walk_length,rand=random.Random(0)):
  '''
  Simulate a random walk starting from start node.
  '''

  walks = []

  gender_choices = [0, 1, 2]
  gender_choices_not_0 = [1, 2]
  gender_choices_not_1 = [0, 2]
  gender_choices_not_2 = [0, 1]

  for walk_iter in range(num_walks):

    for node in G.nodes():
      walk = [node]

      while len(walk) < walk_length:

        cur = walk[-1]
        cur_nbrs = sorted(G.neighbors(cur))
        m_list = []
        f_list = []
        u_list = []
        gender_list = []
        cnt_m = 0
        cnt_f = 0
        cnt_u = 0

        for cur_nbr in cur_nbrs:
          print(G.nodes[cur_nbr]['gender'])
          if G.nodes[cur_nbr]['gender'] == 1:
            cnt_m += 1
            m_list.append(cur_nbr)
          elif G.nodes[cur_nbr]['gender'] == 2:
            cnt_f += 1
            f_list.append(cur_nbr)
          elif G.nodes[cur_nbr]['gender'] == 0:
            cnt_u += 1
            u_list.append(cur_nbr)
        gender_list.append(u_list)
        gender_list.append(m_list)
        gender_list.append(f_list)
        # print(gender_list)
        # print(np.shape(gender_list))
        # print(cnt_f,cnt_m,cnt_u)
        if (cnt_m == 0 and cnt_f == 0):
          gender_choice = 0
        elif (cnt_m == 0 and cnt_u == 0):
          gender_choice = 2
        elif (cnt_f == 0 and cnt_u == 0):
          gender_choice = 1
        elif (cnt_m == 0 and cnt_f != 0 and cnt_u != 0):
          gender_choice = random.choice(gender_choices_not_1)
        elif (cnt_m != 0 and cnt_f == 0 and cnt_u != 0):
          gender_choice = random.choice(gender_choices_not_2)
        elif (cnt_m != 0 and cnt_f != 0 and cnt_u == 0):
          gender_choice = random.choice(gender_choices_not_0)
        else:
          gender_choice = random.choice(gender_choices)
        # print(gender_choice)
        # print(gender_list[gender_choice])
        next = random.choice(gender_list[gender_choice])
        walk.append(next)
      walks.append(walk)

  return walks









def build_deepwalk_corpus1(G, num_paths, path_length, alpha=0,
                          rand=random.Random(0)):
  walks = []

  nodes = list(G.nodes())

  for cnt in range(num_paths):
    rand.shuffle(nodes)
    for node in nodes:
      walks.append(G.random_walk1(path_length, rand=rand, alpha=alpha, start=node))

  return walks

def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0,
                      rand=random.Random(0)):
  walks = []

  nodes = list(G.nodes())

  for cnt in range(num_paths):
    rand.shuffle(nodes)
    for node in nodes:
      yield G.random_walk(path_length, rand=rand, alpha=alpha, start=node)


def clique(size):
    return from_adjlist(permutations(range(1,size+1)))


# http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def parse_adjacencylist(f):
  adjlist = []
  for l in f:
    if l and l[0] != "#":
      introw = [int(x) for x in l.strip().split()]
      row = [introw[0]]
      row.extend(set(sorted(introw[1:])))
      adjlist.extend([row])
  
  return adjlist

def parse_adjacencylist_unchecked(f):
  adjlist = []
  for l in f:
    if l and l[0] != "#":
      adjlist.extend([[int(x) for x in l.strip().split()]])
  
  return adjlist

def load_adjacencylist(file_, undirected=False, chunksize=10000, unchecked=True):

  if unchecked:
    parse_func = parse_adjacencylist_unchecked
    convert_func = from_adjlist_unchecked
  else:
    parse_func = parse_adjacencylist
    convert_func = from_adjlist

  adjlist = []

  t0 = time()
  
  total = 0 
  with open(file_) as f:
    for idx, adj_chunk in enumerate(map(parse_func, grouper(int(chunksize), f))):
      adjlist.extend(adj_chunk)
      total += len(adj_chunk)
  
  t1 = time()

  logger.info('Parsed {} edges with {} chunks in {}s'.format(total, idx, t1-t0))

  t0 = time()
  G = convert_func(adjlist)
  t1 = time()

  logger.info('Converted edges to graph in {}s'.format(t1-t0))

  if undirected:
    t0 = time()
    G = G.make_undirected()
    t1 = time()
    logger.info('Made graph undirected in {}s'.format(t1-t0))

  return G 


def load_edgelist(file_, undirected=True):
  G = Graph()
  with open(file_) as f:
    for l in f:
      x, y = l.strip().split()[:2]
      x = int(x)
      y = int(y)
      G[x].append(y)
      if undirected:
        G[y].append(x)
  
  G.make_consistent()
  return G


def load_matfile(file_, variable_name="network", undirected=True):
  mat_varables = loadmat(file_)
  mat_matrix = mat_varables[variable_name]

  return from_numpy(mat_matrix, undirected)


def from_networkx(G_input, undirected=True):
    G = Graph()

    for idx, x in enumerate(G_input.nodes()):
        for y in iterkeys(G_input[x]):
            G[x].append(y)

    if undirected:
        G.make_undirected()

    return G


def from_numpy(x, undirected=True):
    G = Graph()

    if issparse(x):
        cx = x.tocoo()
        for i,j,v in zip(cx.row, cx.col, cx.data):
            G[i].append(j)
    else:
      raise Exception("Dense matrices not yet supported.")

    if undirected:
        G.make_undirected()

    G.make_consistent()
    return G


def from_adjlist(adjlist):
    G = Graph()
    
    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = list(sorted(set(neighbors)))

    return G


def from_adjlist_unchecked(adjlist):
    G = Graph()
    
    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = neighbors

    return G


