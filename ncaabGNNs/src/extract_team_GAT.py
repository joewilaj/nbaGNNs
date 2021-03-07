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


        defense_index = tf.math.add(inputs[0],tf.constant([self.N + 1,self.N + 1],dtype = tf.int64))

        off = tf.gather(inputs[1],inputs[0], axis=1)
        defense = tf.gather(inputs[1],defense_index, axis=1)

        game_vec = tf.concat([off,defense],axis = 2)


        return game_vec


class To_Sparse(Layer):

    def __init__(self,):
        super(To_Sparse,self).__init__()

    def call(self,inputs):


        sparse_t = tf.sparse.from_dense(inputs[0], name = 'adj_mat')


        return sparse_t
