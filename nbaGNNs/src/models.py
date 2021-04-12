import argparse, logging
import numpy as np
import networkx as nx
import node2vec
import node2vec_stack
import graph
import construct_from_data
import scipy.io
import pandas as pd
import pickle
import sys
import os
import tensorflow as tf
import keras
import warnings
import pdb
import requests
import datetime
import re
import matplotlib.pyplot as plt
import utils_data
import extract_team_GAT
import spektral
import tensorboard





from tensorflow.keras import backend as K
from tensorflow.keras import layers, initializers
from keras.engine.topology import Layer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Input, Lambda, Concatenate, Dropout, ReLU, Reshape,Flatten
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from datetime import date,timedelta




np.set_printoptions(threshold=sys.maxsize)
 

#Diffusion Convolution Network:

#Atwood, Towsley, Diffusion Convolutional Neural Network, November 15, 2015
#arXiv:1511.02136v6 [cs.LG]


#Convolution operations performed when constructing inputs in nba_DCNN_train() and nba_DCNN_test()

def nba_DCNN(height,node2vec_dim): 


    inputs = Input(shape=(6*height*node2vec_dim,))
    line_input = Input(shape = (1,))
    last_5_input = Input(shape = (10,))

    one_hot_input = Input(shape=(60,))

    game_in = Concatenate()([inputs,last_5_input,one_hot_input])

    dense1 = Dense(int(np.floor(6*node2vec_dim*height)),activation = 'tanh')(game_in)
    drop1 = Dropout(.6)(dense1)

    dense2 = Dense(int(np.floor(height*node2vec_dim)))(drop1)
    drop2 = Dropout(.3)(dense2)

    dense3 = Dense(int(np.floor(height*node2vec_dim/4)),activation = 'tanh')(drop2)
    drop3 = Dropout(.2)(dense3)


    add_line_totals = Concatenate()([drop3,line_input,last_5_input])
    prediction = Dense(1)(add_line_totals)

    #pdb.set_trace()

    model = Model(inputs = [inputs,line_input,last_5_input,one_hot_input], outputs = prediction)

    return model




#General Graph Neural Network

#Leskovec, Ying, You. 2020.Design Space for Graph Neural Networks. NeurIPS 2020
#arXiv:2011.08843v1 

#implemented with spektral: https://github.com/danielegrattarola/spektral



def nba_gen(node2vec_dim):

    channels = 40                                    


    node2vec_input = Input(shape=(62,node2vec_dim))  
    node2vec_Veg_input = Input(shape=(31,node2vec_dim))
    A_input = Input(shape=(62,62))
    A_Veg_input = Input(shape=(31,31))

    A_input_sp = extract_team_GAT.To_Sparse()(A_input)
    A_Veg_input_sp = extract_team_GAT.To_Sparse()(A_Veg_input)

    team_inputs = Input(shape=(2,),dtype = tf.int64)
    line_input = Input(shape=(1,))
    last_5_input = Input(shape = (10,))
    one_hot_input = Input(shape=(60,))


    conv = spektral.layers.GeneralConv(channels= channels, batch_norm=True, dropout=0.0, aggregate='mean', activation='elu', use_bias=True,
                                kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
                                bias_regularizer=None, activity_regularizer=None, 
                                kernel_constraint=None, bias_constraint=None)([node2vec_input,A_input_sp])



    conv_veg = spektral.layers.GeneralConv(channels= channels, batch_norm=True, dropout=0.0, aggregate='mean', activation='elu', use_bias=True,
                                kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
                                bias_regularizer=None, activity_regularizer=None, 
                                kernel_constraint=None, bias_constraint=None)([node2vec_Veg_input,A_Veg_input_sp])


    #extracts nodes for link prediction

    game_vec = extract_team_GAT.Game_Vec(channels)([team_inputs,conv,conv_veg])


    rshp = Reshape((int(np.floor(6*channels)),))(game_vec)
    cat = Concatenate()([rshp,one_hot_input])



    dense1 = Dense(int(np.floor(6.5*channels)),activation = 'tanh')(cat)
    drop1 = Dropout(.01)(dense1)

    dense2 = Dense(int(np.floor(2*channels)),activation = 'tanh')(drop1)
    drop2 = Dropout(.01)(dense2)

    drop2 = Concatenate()([drop2,last_5_input])

    dense3 = Dense(int(np.floor(channels/2)))(drop2)
    drop3 = Dropout(.01)(dense3)

    add_line = Concatenate()([drop3,line_input])

    prediction = Dense(1)(add_line)


    model = Model(inputs = [team_inputs,line_input,node2vec_input,A_input,node2vec_Veg_input,A_Veg_input,last_5_input,one_hot_input], outputs = prediction)

    return model


    #ARMA model
 
    #Filippo Maria Bianchi, Daniele Grattarola, Lorenzo Livi, Cesare Alippi
    #Graph Neural Networks with convolutional ARMA filters,January 15,2019
    #arXiv:1901.01343v7 [cs.LG] 

    #implemented with spektral: https://github.com/danielegrattarola/spektral




def nba_ARMA(node2vec_dim):

    channels = 30                                                         


    node2vec_input = Input(shape=(62,node2vec_dim))  
    node2vec_Veg_input = Input(shape=(31,node2vec_dim))
    A_input = Input(shape=(62,62))
    A_Veg_input = Input(shape=(31,31))

    team_inputs = Input(shape=(2,),dtype = tf.int64)
    line_input = Input(shape=(1,))
    last_5_input = Input(shape = (10,))
    one_hot_input = Input(shape=(60,))




    ARMA = spektral.layers.ARMAConv(channels, order=4, iterations=1, share_weights=False, gcn_activation='relu', 
                            dropout_rate=0.2, activation='elu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                            kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
                            kernel_constraint=None, bias_constraint=None)([node2vec_input,A_input])



    ARMA_Veg = spektral.layers.ARMAConv(channels, order=4, iterations=1, share_weights=False, gcn_activation='relu', 
                            dropout_rate=0.2, activation='elu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                            kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
                            kernel_constraint=None, bias_constraint=None)([node2vec_Veg_input,A_Veg_input])


    #extracts nodes for link prediction


    game_vec = extract_team_GAT.Game_Vec(channels)([team_inputs,ARMA,ARMA_Veg])

    rshp = Reshape((int(np.floor(6*channels)),))(game_vec)
    cat = Concatenate()([rshp,one_hot_input])



    dense1 = Dense(int(np.floor(6.5*channels)),activation = 'tanh')(cat)
    drop1 = Dropout(.05)(dense1)

    dense2 = Dense(int(np.floor(2*channels)),activation = 'tanh')(drop1)
    drop2 = Dropout(.05)(dense2)

    drop2 = Concatenate()([drop2,last_5_input])

    dense3 = Dense(int(np.floor(channels/2)))(drop2)
    drop3 = Dropout(.05)(dense3)

    add_line = Concatenate()([drop3,line_input])

    prediction = Dense(1)(add_line)

    model = Model(inputs = [team_inputs,line_input,node2vec_input,A_input,node2vec_Veg_input,A_Veg_input,last_5_input,one_hot_input], outputs = prediction)

    return model


def nba_gin(node2vec_dim):

    channels = 40                                                    


    node2vec_input = Input(shape=(62,node2vec_dim))  
    node2vec_Veg_input = Input(shape=(31,node2vec_dim))
    A_input = Input(shape=(62,62))
    A_Veg_input = Input(shape=(31,31))

    team_inputs = Input(shape=(2,),dtype = tf.int64)
    line_input = Input(shape=(1,))
    last_5_input = Input(shape = (10,))
    one_hot_input = Input(shape=(60,))

    A_input_sp = extract_team_GAT.To_Sparse()(A_input)
    A_Veg_input_sp = extract_team_GAT.To_Sparse()(A_Veg_input)





    GIN = spektral.layers.GINConv(channels, epsilon=None, mlp_hidden=[channels, channels], mlp_activation='relu', aggregate='sum', activation= None, 
                                  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
                                  bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                                  bias_constraint=None)([node2vec_input,A_input_sp])



    GIN_Veg = spektral.layers.GINConv(channels, epsilon=None, mlp_hidden=[channels, channels], mlp_activation='relu', aggregate='sum', activation= None, 
                                  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
                                  bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                                  bias_constraint=None)([node2vec_Veg_input,A_Veg_input_sp])

    game_vec = extract_team_GAT.Game_Vec(channels)([team_inputs,GIN,GIN_Veg])



    rshp = Reshape((int(np.floor(6*channels)),))(game_vec)
    cat = Concatenate()([rshp,one_hot_input])



    dense1 = Dense(int(np.floor(6.5*channels)),activation = 'tanh')(cat)
    drop1 = Dropout(.01)(dense1)

    dense2 = Dense(int(np.floor(2*channels)),activation = 'tanh')(drop1)
    drop2 = Dropout(.01)(dense2)

    drop2 = Concatenate()([drop2,last_5_input])

    dense3 = Dense(int(np.floor(channels/2)))(drop2)
    drop3 = Dropout(.01)(dense3)

    add_line = Concatenate()([drop3,line_input])

    prediction = Dense(1)(add_line)

    #extracts nodes for link prediction


    model = Model(inputs = [team_inputs,line_input,node2vec_input,A_input,node2vec_Veg_input,A_Veg_input,last_5_input,one_hot_input], outputs = prediction)

    return model



def discriminator(node2vec_dim):

    channels = 40                                                   


    feature_input = Input(shape=(62,node2vec_dim))  
    feature_Veg_input = Input(shape=(31,node2vec_dim))
    feature_M_input = Input(shape=(31,node2vec_dim))
    A_input = Input(shape=(62,62))
    A_Veg_input = Input(shape=(31,31))
    M_Graph_input = Input(shape=(31,31))

    team_inputs = Input(shape=(2,),dtype = tf.int64)
    line_input = Input(shape=(1,))
    model_input = Input(shape=(1,))
    last_5_input = Input(shape = (10,))
    one_hot_input = Input(shape=(60,))

    A_input_sp = extract_team_GAT.To_Sparse()(A_input)
    A_Veg_input_sp = extract_team_GAT.To_Sparse()(A_Veg_input)
    M_Graph_input_sp = extract_team_GAT.To_Sparse()(M_Graph_input)





    GIN = spektral.layers.GINConv(channels, epsilon=None, mlp_hidden=[channels, channels], mlp_activation='relu', aggregate='sum', activation= None, 
                                  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
                                  bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                                  bias_constraint=None)([feature_input,A_input_sp])



    GIN_Veg = spektral.layers.GINConv(channels, epsilon=None, mlp_hidden=[channels, channels], mlp_activation='relu', aggregate='sum', activation= None, 
                                  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
                                  bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                                  bias_constraint=None)([feature_Veg_input,A_Veg_input_sp])


    GIN_M = spektral.layers.GINConv(channels, epsilon=None, mlp_hidden=[channels, channels], mlp_activation='relu', aggregate='sum', activation= None, 
                                  use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
                                  bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                                  bias_constraint=None)([feature_M_input,M_Graph_input_sp])

    game_vec = extract_team_GAT.Game_Vec_D(channels)([team_inputs,GIN,GIN_Veg,GIN_M])



    rshp = Reshape((int(np.floor(8*channels)),))(game_vec)
    cat = Concatenate()([rshp,one_hot_input,line_input,model_input])



    dense1 = Dense(int(np.floor(8.5*channels)),activation = 'tanh')(cat)
    drop1 = Dropout(.01)(dense1)

    dense2 = Dense(int(np.floor(4*channels)),activation = 'tanh')(drop1)
    drop2 = Dropout(.01)(dense2)

    drop2 = Concatenate()([drop2,last_5_input])

    dense3 = Dense(int(np.floor(channels/2)))(drop2)
    drop3 = Dropout(.01)(dense3)

    

    prediction = Dense(2,activation = 'softmax')(drop3)



    #extracts nodes for link prediction


    model = Model(inputs = [team_inputs,line_input,model_input,feature_input,A_input,feature_Veg_input,A_Veg_input,feature_M_input,
               M_Graph_input,last_5_input,one_hot_input], outputs = prediction)

    return model



def main():

    #select model type and year


    
    #model_type = 'nba_DCNN'
    #model_type = 'nba_gen'
    #model_type = 'nba_ARMA'
    model_type = 'nba_gin'

    year = 2021

    #select day range on which to test the model

    startdate = datetime.datetime(year,3,1)
    stopdate = datetime.datetime(year,4,12)



    #plots = 'on'
    plots = 'off'





    start_day = (startdate-datetime.datetime(year-1,10,12)).days
    stop_day = (stopdate-datetime.datetime(year-1,10,12)).days

    startstring = startdate.strftime("%m_%d_%Y")
    stopstring = stopdate.strftime("%m_%d_%Y")

    now = datetime.datetime.now()
    datestring = now.strftime("%m_%d_%Y")

    today = (now-datetime.datetime(year-1,10,12)).days


    TeamLists = pd.read_excel('data/TeamLists.xls',sheet_name = 0,header = None)
    TeamLists = TeamLists.to_numpy(dtype = object,copy = True)

    if year > 2014:
        TeamList = TeamLists[:,2]

    elif year == 2014:
        TeamList = TeamLists[:,1] 

    elif year < 2014:
        TeamList = TeamLists[:,0]


    TeamList_Lines = TeamLists[:,3]



    #edgeweights when constructing the Offense and Defense Statistic graphs SOffDef, G_orc

    weights = pd.read_excel('data/weights.xls',sheet_name = 0,header = 0)
    weights = weights.to_numpy(dtype = object,copy = True)





    with open('pickles/NBA_Data_pickled/'+str(year)+'NBAData.pkl', 'rb') as Data: 
        Data_Full = pickle.load(Data)



    schedule, HomeAway = utils_data.format_schedule(Data_Full,TeamList,year)


    #Lines obtained from https://www.kaggle.com/erichqiu/nba-odds-and-scores


    if year < 2021:
        Lines = utils_data.Lines(Data_Full,schedule,HomeAway,TeamList_Lines,year)


    if year == 2021:
        Lines = utils_data.Lines_2021(Data_Full,schedule,HomeAway,TeamList_Lines,year)



    ats_bets = 0
    ats_wins = 0
    total_bets = 0
    total_wins = 0
    money_line_wins = 0
    moneyline_count = 0
    window = 1  #parameter to constrain the test set to games where the model prediction and vegas prediction differ more than 'window'
    push = 0
    ties = 0

    loss = 0
    runs = 0


    test_games_all = np.zeros((1300,8),dtype = object)
    test_count = 0

    #For each day a game occurs, the model constructs a training and validation set using all games played previously in the season
    #The model is tested on games occuring the current day     


    for day in range(start_day,stop_day):



        if np.sum(schedule[:,day+1]) != -30:
            
            runs = runs + 1

            #Construct S_oracle and Pts


            #Graph constructed according to:

            #Balreira, Miceli, Tegtmeyer,  An Oracle method to predict NFL games,
            #http://ramanujan.math.trinity.edu/bmiceli/research/NFLRankings_revised_print.pdf

            #using data from https://github.com/roclark/sportsipy 

            #Vegas Graphs are constructed using the vegas lines to construct the a point differential graph
            #using data from https://www.kaggle.com/erichqiu/nba-odds-and-scores


            S_OffDef, A_OffDef = construct_from_data.construct_S_orc(Data_Full,schedule,HomeAway,weights,day)




            Vegas_Graph = construct_from_data.Vegas_Graph(schedule,Lines,day)
            A_Veg = A_OffDef[0:31,31:62]

            ARMA = spektral.utils.convolution.normalized_laplacian(A_OffDef)
            ARMA = spektral.utils.convolution.rescale_laplacian(ARMA)

            ARMA_Veg = spektral.utils.convolution.normalized_laplacian(A_Veg)
            ARMA_Veg = spektral.utils.convolution.rescale_laplacian(ARMA_Veg)


            epsilon = .001 #hyperparameter to perform PageRank

            #hyperparameters for node2vec
            #Grover, Leskovec, node2vec: Scalable Feature Learning for Networks, July 3, 2016 #arXiv:1607.00653v1 [cs.SI]

            node2vec_dim = 20
            node2vec_p = 1
            node2vec_q = 1  # q < 1 DFS (homophily)   or  q > 1 BFS (structural equivalence)

            height = 3
            n2v_walklen = 6
            n2v_numwalks = 20
            n2v_wsize = 7
            n2v_iter = 1
            n2v_workers = 8



            if model_type == 'nba_DCNN' or model_type == 'nba_gen' or model_type == 'nba_ARMA' or model_type == 'nba_gin':

                G_orc = (1-epsilon)*(S_OffDef) + epsilon*(1/62)*np.ones((62,62),dtype = float)
                G_orc = utils_data.sto_mat(G_orc)

                PageRank_Off, PageRank_Def = construct_from_data.PageRank(G_orc)

                args_N = node2vec_stack.node2vec_input(S_OffDef,'emb/NBA'+str(year)+'node2vec_OffDef.txt',node2vec_dim,n2v_walklen,
                                                            n2v_numwalks,n2v_wsize,n2v_iter,n2v_workers,node2vec_p,node2vec_q,True,True,False,False)

                featurevecs = node2vec_stack.feat_N(args_N)

                feature_node2vec = np.zeros((62,node2vec_dim),dtype = float)

                for j in range(62):
                    feature_node2vec[j,:] = featurevecs[str(j)]


                args_N_Veg = node2vec_stack.node2vec_input(Vegas_Graph,'emb/node2vec_Veg.txt',node2vec_dim,n2v_walklen,
                                                            n2v_numwalks,n2v_wsize,n2v_iter,n2v_workers,node2vec_p,node2vec_q,True,True,False,False)

                featurevecs_Veg = node2vec_stack.feat_N(args_N_Veg)

                feature_node2vec_Veg = np.zeros((31,node2vec_dim),dtype = float)

                for j in range(31):
                    feature_node2vec_Veg[j,:] = featurevecs_Veg[str(j)]

                if plots == 'on':
                    utils_data.plot_node2vec(feature_node2vec_Veg,TeamList_Lines,PageRank_Off,PageRank_Def,Vegas_Graph)
                

                if model_type == 'nba_DCNN':
    
    
                    S_OffDef_stack = np.zeros((62,62,height),dtype = float)
                    Vegas_Graph_stack = np.zeros((31,31,height),dtype = float)

                    for j in range(height):
                        S_OffDef_stack[:,:,j] = np.linalg.matrix_power(S_OffDef,j+1)
                        Vegas_Graph_stack[:,:,j] = np.linalg.matrix_power(Vegas_Graph,j+1)



                    x_train, y_train, line_train,last_5_train, one_hot_train = construct_from_data.Training_Set_nba_DCNN(Data_Full,Lines,schedule,HomeAway,day,
                                                                                                S_OffDef_stack,Vegas_Graph_stack,feature_node2vec,feature_node2vec_Veg,height,node2vec_dim)


                elif model_type == 'nba_gen':



                    x_train, y_train, line_train,feature_train,A_Train,feature_Veg_train,A_Veg_train,last_5_train,one_hot_train = construct_from_data.GAT_training_set(Data_Full,
                                                                                                                    Lines,schedule,HomeAway,
                                                                                                                    day,feature_node2vec,
                                                                                                                    A_OffDef,feature_node2vec_Veg,A_Veg)

                    

                elif model_type == 'nba_ARMA':

                    x_train, y_train, line_train,feature_train,A_Train,feature_Veg_train,A_Veg_train, last_5_train,one_hot_train = construct_from_data.GAT_training_set(Data_Full,
                                                                                                                    Lines,schedule,HomeAway,
                                                                                                                    day,feature_node2vec,
                                                                                                                    ARMA,feature_node2vec_Veg,ARMA_Veg)

                elif model_type == 'nba_gin':

                    x_train, y_train, line_train,feature_train,A_Train,feature_Veg_train,A_Veg_train, last_5_train,one_hot_train = construct_from_data.gin_training_set(Data_Full,
                                                                                                                    Lines,schedule,HomeAway,
                                                                                                                    day,feature_node2vec,
                                                                                                                    ARMA,feature_node2vec_Veg,ARMA_Veg)



            call_backs = EarlyStopping(monitor='val_loss', min_delta=0, patience=150, verbose=1, restore_best_weights= True)

            #Train the model on all previous games

            #opt = SGD(lr = .001)   
            opt = Adam(learning_rate=0.001)         

            if model_type == 'nba_DCNN':
                model = nba_DCNN(height,node2vec_dim)
                model.compile(loss='mean_squared_error', optimizer= opt, metrics=['accuracy'])
                model.fit([x_train,line_train,last_5_train,one_hot_train],y_train, 
                            epochs = 20, batch_size = 15, validation_split = 0.05,callbacks = [call_backs]) 
            
                model.summary()

            elif model_type == 'nba_gen':
                model = nba_gen(node2vec_dim)
                model.compile(loss='mean_squared_error', optimizer= opt, metrics=['accuracy'])
                model.fit([x_train,line_train,feature_train,A_Train,feature_Veg_train,A_Veg_train,last_5_train,one_hot_train],y_train, 
                            epochs = 10,batch_size = 1,validation_split = 0.05,callbacks = [call_backs])
                model.summary()

            elif model_type == 'nba_ARMA':
                model = nba_ARMA(node2vec_dim)
                model.compile(loss='mean_squared_error', optimizer= opt, metrics=['accuracy'])
                model.fit([x_train,line_train,feature_train,A_Train,feature_Veg_train,A_Veg_train,last_5_train,one_hot_train],y_train, 
                            epochs = 10,batch_size = 1,validation_split = 0.05,callbacks = [call_backs])
                model.summary()

            elif model_type == 'nba_gin':
                model = nba_gin(node2vec_dim)
                model.compile(loss='mean_squared_error', optimizer= opt, metrics=['accuracy'])
                model.fit([x_train,line_train,feature_train,A_Train,feature_Veg_train,A_Veg_train,last_5_train,one_hot_train],y_train, 
                            epochs = 10,batch_size = 1,validation_split = 0.05,callbacks = [call_backs])
                model.summary()



            
            games, gameteams, testgamecount = construct_from_data.Test_Games(TeamList,Data_Full,schedule,HomeAway,Lines,day)



            
            if model_type == 'nba_DCNN':
                x_test, line_test, last_5_test, test_y,one_hot_test = construct_from_data.Test_Set_nba_DCNN(Data_Full,games,testgamecount,S_OffDef_stack,Vegas_Graph_stack,
                                                                            feature_node2vec,feature_node2vec_Veg,height,node2vec_dim,day,year)

                Pred = model.predict([x_test,line_test,last_5_test,one_hot_test])

                if year < 2021:
                    Eval = model.evaluate([x_test,line_test,last_5_test,one_hot_test],test_y,verbose=0)
                    loss = loss + Eval[0]

            #test the model, print predictions, the ATS Win %, ML Win % and the MSE for the test set

            elif model_type == 'nba_gen':

                

                x_test, line_test,feature_test,A_test,feature_Veg_test,A_Veg_test,last_5_test, test_y,one_hot_test = construct_from_data.GAT_test_set(Data_Full,games,
                                                                                                                testgamecount,feature_node2vec,
                                                                                                                A_OffDef,feature_node2vec_Veg,A_Veg,day,year)

                Pred = model.predict([x_test,line_test,feature_test,A_test,feature_Veg_test,A_Veg_test,last_5_test,one_hot_test],batch_size=1)
                if year < 2021:
                    Eval = model.evaluate([x_test,line_test,feature_test,A_test,feature_Veg_test,A_Veg_test,last_5_test,one_hot_test],test_y,verbose=0,batch_size=1)
                    loss = loss + Eval[0]


            elif model_type == 'nba_ARMA':
                x_test, line_test,feature_test,A_test,feature_Veg_test,A_Veg_test,last_5_test, test_y,one_hot_test = construct_from_data.GAT_test_set(Data_Full,games,
                                                                                                                testgamecount,feature_node2vec,
                                                                                                                ARMA,feature_node2vec_Veg,ARMA_Veg,day,year)

                Pred = model.predict([x_test,line_test,feature_test,A_test,feature_Veg_test,A_Veg_test,last_5_test,one_hot_test],batch_size=1)
                if year < 2021:
                    Eval = model.evaluate([x_test,line_test,feature_test,A_test,feature_Veg_test,A_Veg_test,last_5_test,one_hot_test],test_y,verbose=0,batch_size=1)
                    loss = loss + Eval[0]


            elif model_type == 'nba_gin':

                

                x_test, line_test,feature_test,A_test,feature_Veg_test,A_Veg_test,last_5_test, test_y,one_hot_test = construct_from_data.gin_test_set(Data_Full,games,
                                                                                                                testgamecount,feature_node2vec,
                                                                                                                A_OffDef,feature_node2vec_Veg,A_Veg,day,year)

                Pred = model.predict([x_test,line_test,feature_test,A_test,feature_Veg_test,A_Veg_test,last_5_test,one_hot_test],batch_size=1)
                if year < 2021:
                    Eval = model.evaluate([x_test,line_test,feature_test,A_test,feature_Veg_test,A_Veg_test,last_5_test,one_hot_test],test_y,verbose=0,batch_size=1)
                    loss = loss + Eval[0]



            results= np.round(Pred,decimals = 1)
            games = np.concatenate((games,Pred),axis = 1)

            if day != stop_day-1:

                test_count = test_count + games.shape[0]


                test_games_all[(test_count - games.shape[0]):test_count,:] = games




            gameteams = np.concatenate((gameteams,results),axis = 1)


            df = pd.DataFrame(gameteams)
            df.style.set_properties(**{'text-align': 'left'})
            df1 = df.to_string(index=False,header = False)

            print(df1)

            if today == day+1:
                if model_type == 'nba_DCNN':
                    df.to_excel('predictions/'+datestring+'_DCNN_predictions.xls', header = ['Home','Away',model_type + ' prediction'],index=False)

                if model_type == 'nba_gen':
                    df.to_excel('predictions/'+datestring+'_GEN_predictions.xls', header = ['Home','Away',model_type + ' prediction'],index=False)

                if model_type == 'nba_ARMA':
                    df.to_excel('predictions/'+datestring+'_ARMA_predictions.xls', header = ['Home','Away',model_type + ' prediction'],index=False)

                if model_type == 'nba_gin':
                    df.to_excel('predictions/'+datestring+'_GIN_predictions.xls', header = ['Home','Away',model_type + ' prediction'],index=False)

            
            bet_stats = utils_data.Model_Eval_ML_ATS(games,testgamecount,ats_bets,ats_wins,total_bets,total_wins,
                                                     money_line_wins,moneyline_count,window,push,ties)

            ats_bets = bet_stats[0]
            ats_wins = bet_stats[1]
            money_line_wins = bet_stats[4]
            moneyline_count = bet_stats[5]
            push = bet_stats[7]
            ties = bet_stats[8]


    test_games_all = test_games_all[~np.all(test_games_all == 0, axis=1)]




    print(model_type)
    print(startstring + ' to ' + stopstring)


    #Evaluate model against Vegas


    if (ats_bets - push) != 0 and (moneyline_count - ties) != 0:

        ats_win_percentage = ats_wins/(ats_bets - push)
        moneyline_win_percentage = money_line_wins/(moneyline_count - ties)

        print('ats win %: ' + str(round(ats_win_percentage,3)))
        print('ml win %: ' + str(round(moneyline_win_percentage,3)))
        
        if year < 2021:
            print('MSE: '  + str(round((loss/runs),1)))


    if today != day + 1:
        utils_data.eval_plots(test_games_all,window)



    M_Graph = construct_from_data.Model_Graph(test_games_all,schedule,day)

    args_M = node2vec_stack.node2vec_input(M_Graph,'emb/NBA'+str(year)+'node2vec_OffDef.txt',node2vec_dim,n2v_walklen,
                                                            n2v_numwalks,n2v_wsize,n2v_iter,n2v_workers,node2vec_p,node2vec_q,True,True,False,False)

    featurevecs = node2vec_stack.feat_N(args_M)

    feature_node2vec_M = np.zeros((31,node2vec_dim),dtype = float)

    for j in range(31):
        feature_node2vec_M[j,:] = featurevecs[str(j)]




    vec = construct_from_data.Discriminator_Training_Set_GAT(Data_Full,test_games_all,
                                                         feature_node2vec,A_OffDef,feature_node2vec_Veg,A_Veg,feature_node2vec_M,M_Graph,day)

    x_train = vec[0]
    y_train = vec[1]
    line_train = vec[2]
    model_train = vec[3]
    feature_train = vec[4]
    A_Train = vec[5]
    feature_Veg_train = vec[6]
    A_Veg_train = vec[7]
    feature_M_train = vec[8]
    M_Graph_train = vec[9]
    last_5_train = vec[10]
    one_hot = vec[11]



    model = discriminator(node2vec_dim)
    model.compile(loss='binary_crossentropy', optimizer= opt, metrics=['accuracy'])
    model.fit([x_train,line_train,model_train,feature_train,A_Train,feature_Veg_train,A_Veg_train,feature_M_train,
               M_Graph_train,last_5_train,one_hot],y_train, epochs = 10,batch_size = 1,validation_split = 0.1,callbacks = [call_backs])
    model.summary()


    vec = construct_from_data.Discriminator_Test_Set_GAT(Data_Full,games,
                                                         feature_node2vec,A_OffDef,feature_node2vec_Veg,A_Veg,feature_node2vec_M,M_Graph,day)


    x_test = vec[0]
    line_test = vec[1]
    model_test = vec[2]
    feature_test = vec[3]
    A_Test = vec[4]
    feature_Veg_test = vec[5]
    A_Veg_test = vec[6]
    feature_M_test = vec[7]
    M_Graph_test = vec[8]
    last_5_test = vec[9]
    one_hot = vec[10]

    Pred2 = model.predict([x_test,line_test,model_test,feature_test,A_Test,feature_Veg_test,A_Veg_test,feature_M_test,
                         M_Graph_test,last_5_test,one_hot_test],batch_size=1)

    Pred2 = np.round(Pred2,decimals = 2)

    gameteams = np.concatenate((gameteams,Pred2),axis = 1)
    gameteams = gameteams[:,0:4]

    df = pd.DataFrame(gameteams)
    df.style.set_properties(**{'text-align': 'left'})
    df1 = df.to_string(index=False,header = False)

    print(df1)


    if today == day+1:
        df.to_excel('predictions/'+datestring+'_GIN_predictions.xls', header = ['Home','Away',model_type + ' prediction','model_confidence'],index=False)



if __name__ == "__main__":
    main()

