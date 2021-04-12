from __future__ import absolute_import

from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dropout, LeakyReLU, Dense, Concatenate,Reshape
import tensorflow as tf
import numpy as np
import pdb

#Custom Layer to extract offense and defense nodes for home and away teams

class Game_Vec(Layer):

    def __init__(self,attention_feat_size):
        super(Game_Vec,self).__init__()
        self.feat_dim = attention_feat_size

    def call(self,inputs):


        defense_index = tf.math.add(inputs[0],tf.constant([31,31],dtype = tf.int64))

        off = tf.gather(inputs[1],inputs[0], axis=1)
        defense = tf.gather(inputs[1],defense_index, axis=1)
        vegas = tf.gather(inputs[2],inputs[0], axis=1)

        game_vec = tf.concat([off,defense,vegas],axis = 2)


        return game_vec

class Game_Vec_D(Layer):

    def __init__(self,attention_feat_size):
        super(Game_Vec_D,self).__init__()
        self.feat_dim = attention_feat_size

    def call(self,inputs):


        defense_index = tf.math.add(inputs[0],tf.constant([31,31],dtype = tf.int64))

        off = tf.gather(inputs[1],inputs[0], axis=1)
        defense = tf.gather(inputs[1],defense_index, axis=1)
        vegas = tf.gather(inputs[2],inputs[0], axis=1)
        model = tf.gather(inputs[3],inputs[0], axis=1)

        game_vec = tf.concat([off,defense,vegas,model],axis = 2)


        return game_vec


class To_Sparse(Layer):

    def __init__(self,):
        super(To_Sparse,self).__init__()

    def call(self,inputs):


        sparse_t = tf.sparse.from_dense(inputs[0], name = 'adj_mat')


        return sparse_t

