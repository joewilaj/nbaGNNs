from __future__ import absolute_import

from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dropout, LeakyReLU, Dense, Concatenate,Reshape
import tensorflow as tf
import numpy as np
import pdb

#Custom Layer to extract offense and defense nodes for home and away teams

class Game_Vec(Layer):

    def __init__(self,attention_feat_size,N):
        super(Game_Vec,self).__init__()
        self.feat_dim = attention_feat_size
        self.N = N

    def call(self,inputs):

        index = tf.math.add(inputs[0],tf.constant([0,self.N + 1],dtype = tf.int64))
        stack = tf.gather(inputs[1],index, axis=1)


        return stack


class To_Sparse(Layer):

    def __init__(self,):
        super(To_Sparse,self).__init__()

    def call(self,inputs):


        sparse_t = tf.sparse.from_dense(inputs[0], name = 'adj_mat')


        return sparse_t

