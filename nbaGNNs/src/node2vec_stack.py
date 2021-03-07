
import numpy as np
import argparse, logging
import numpy as np
import networkx as nx
import node2vec
import graph
import scipy.io
import warnings
import pdb


from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


class node2vec_input:
    def __init__(self,ginput,goutput,dimensions,walk_length,num_walks,window_size,iternum,workers,p,q,weighted,directed,unweighted,undirected):
        self.ginput = ginput
        self.goutput = goutput
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.iternum = iternum
        self.workers = workers
        self.p = p
        self.q = q
        self.weighted = weighted
        self.directed = directed
        self.unweighted = unweighted
        self.undirected = undirected



#Implementation of node2vec https://github.com/aditya-grover/node2vec

def read_graph_N(args_N):
    '''
    Reads the input network in networkx.
    '''
    if args_N.weighted:
        #G = nx.read_edgelist(args_N.ginput, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
        G = nx.from_numpy_matrix(args_N.ginput, parallel_edges=False, create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args_N.ginput, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args_N.directed:
        G = G.to_undirected()

    return G


def learn_embeddings_N(walks_N,args_N):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks_N = [list(map(str, walk_N)) for walk_N in walks_N]
    model_N = Word2Vec(walks_N, size=args_N.dimensions, window=args_N.window_size, min_count=0, sg=1, workers=args_N.workers, iter=args_N.iternum)
    model_N.wv.save_word2vec_format(args_N.goutput)
    featurevecs = model_N.wv

    return featurevecs

def feat_N(args_N):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    nx_G = read_graph_N(args_N)
    G = node2vec.Graph_N(nx_G, args_N.directed, args_N.p, args_N.q)
    G.preprocess_transition_probs()
    walks_N = G.simulate_walks(args_N.num_walks, args_N.walk_length)
    featurevecs = learn_embeddings_N(walks_N,args_N)
    
    return featurevecs