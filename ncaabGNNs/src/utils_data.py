import argparse, logging
import numpy as np
import networkx as nx
import node2vec
import graph
import construct_from_data
import scipy.io
import pandas as pd
import pickle
import sys
import tensorflow as tf
import keras
import warnings
import pdb
import matplotlib
import matplotlib.pyplot as plt
import datetime



from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from keras import backend as K
from keras import layers
from keras.engine.topology import Layer
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Lambda, Concatenate, Dropout, ReLU
from keras.optimizers import SGD
from keras.utils import plot_model
from mpl_toolkits.mplot3d import Axes3D
from random import seed
from random import randint
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.text import Annotation
from matplotlib import colors
from sklearn.cluster import KMeans




class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz        

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)


def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)




def sto_mat(A):

    numrows,numcols = A.shape
    n = numrows

    for j in range(n):
        if np.sum(A[j,:]) != 0:
            A[j,:] = (1/np.sum(A[j,:]))*A[j,:]

    for j in range(n):
        if np.sum(A[j,:]) < 1 and np.sum(A[j,:]) > 0:
            A[j,n-1] = A[j,n-1] + (1 - np.sum(A[j,:]))
        elif np.sum(A[j,:]) > 1:
            A[j,n-1] = A[j,n-1] - (np.sum(A[j,:])-1)
        


    for j in range(n):
        if np.sum(A[j,:]) < 1 and np.sum(A[j,:]) > 0:
            A[j,n-1] = A[j,n-1] + (1 - np.sum(A[j,:]))
        elif np.sum(A[j,:]) > 1:
            A[j,n-1] = A[j,n-1] - (np.sum(A[j,:])-1)

    return A





    return Lines, Totals, schedule, HomeAway


def format_schedule(Data_Full,TeamList,year):

    schedule = np.ones((TeamList.shape[0],364),dtype = int)
    schedule  = -1*schedule
    HomeAway = np.zeros((TeamList.shape[0],364),dtype = int)



    for k in range(TeamList.shape[0]):
        for n in range(364):
            p = 0
            for m in range(TeamList.shape[0]):
                if Data_Full[n,7,k] == TeamList[m]:
                    schedule[k,n] = m
                    p = p+1
                    break

            if Data_Full[n,6,k] == 'Away' and p ==1:
                HomeAway[k,n] = 1
            elif Data_Full[n,6,k] == 'Home' and p == 1:
                HomeAway[k,n] = 2
            elif Data_Full[n,6,k] == 'Neutral' and p == 1:
                if k < schedule[k,n]:
                    HomeAway[k,n] = 2
                else:
                    HomeAway[k,n] = 1


    return schedule,HomeAway



def Model_Eval_ML_ATS(games,testgamecount,ats_bets,ats_wins,total_bets,total_wins,money_line_wins,moneyline_count,window,push,ties):

    for z in range(testgamecount):

        if (games[z,4] !=0 or games[z,5] !=0) and (games[z,4] is not None):
            moneyline_count = moneyline_count + 1

            if games[z,4]-games[z,5] > 0 and games[z,7] > 0:
                money_line_wins = money_line_wins + 1
            elif games[z,4]-games[z,5] < 0 and games[z,7] < 0:
                money_line_wins = money_line_wins + 1
            elif games[z,4]-games[z,5] == 0:
                ties = ties + 1

    return [ats_bets, ats_wins, total_bets, total_wins, money_line_wins, moneyline_count, window, push, ties];




def plot_node2vec(feature_node2vec,TeamList,PageRank_Off,PageRank_Def,Vegas_Graph):

    TeamList_D = np.zeros((TeamList.shape[0]+1,),dtype = object)

    TeamList = np.concatenate((TeamList,['Oracle Offense']),axis = -1)
    TeamList_D[0:TeamList.shape[0]+1] = TeamList



    for i in range(TeamList.shape[0]):
        TeamList_D[i] = TeamList_D[i] + ' Defense'

    TeamList_D[TeamList.shape[0]] = 'Oracle Defense'

    Labels = np.concatenate((TeamList,TeamList_D),axis = -1)

    Labels_Veg = Labels[0:31]

    X = np.concatenate((np.transpose([Labels_Veg]),feature_node2vec),axis = -1)

    kmeans = KMeans(n_clusters=8, random_state=0).fit(feature_node2vec)
    labs = kmeans.labels_ 


    #pdb.set_trace()

    np.random.seed(19680801)

    p = randint(1,feature_node2vec.shape[1])
    q = randint(1,feature_node2vec.shape[1])
    r = randint(1,feature_node2vec.shape[1])


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:,p], X[:,q], X[:,r],c = labs)

    xyzn = zip(X[:,p],X[:,q], X[:,r])

    for j, xyz_ in enumerate(xyzn): 
        annotate3D(ax, s= Labels[j], xyz=xyz_, fontsize=10, xytext=(-3,3),
               textcoords='offset points', ha='right',va='bottom')

    plt.show()












def eval_plots(test_games_all,window):

    freq_count_wins = np.zeros((TeamList.shape[0],),dtype = float)
    freq_count_losses = np.zeros((TeamList.shape[0],),dtype = float)

    for i in range(test_games_all.shape[0]):

        if (test_games_all[i,4] !=0 or test_games_all[i,5] !=0) and (test_games_all[i,4] is not None):

            if abs(((test_games_all[i,4]-test_games_all[i,5])-test_games_all[i,2])) > window:
                pos = int(np.floor(test_games_all[i,2]))

                if test_games_all[i,4]-test_games_all[i,5] > test_games_all[i,2] and test_games_all[i,7] > test_games_all[i,2]:

                    if pos >= 15:
                        freq_count_wins[29] = freq_count_wins[29] + 1

                    elif pos <= -15:
                        freq_count_wins[0] = freq_count_wins[0] + 1

                    elif abs(pos) <= 14:
                        freq_count_wins[pos+14] = freq_count_wins[pos+14] + 1


                elif test_games_all[i,4]-test_games_all[i,5] < test_games_all[i,2] and test_games_all[i,7] < test_games_all[i,2]:

                    pos = -1*pos

                    if pos >= 15:
                        freq_count_wins[29] = freq_count_wins[29] + 1

                    elif pos <= -15:
                        freq_count_wins[0] = freq_count_wins[0] + 1

                    elif abs(pos) <= 14:
                        freq_count_wins[pos+14] = freq_count_wins[pos+14] + 1


                elif test_games_all[i,4]-test_games_all[i,5] == test_games_all[i,2]:
                    pass


                elif test_games_all[i,4]-test_games_all[i,5] < test_games_all[i,2] and test_games_all[i,7] > test_games_all[i,2]:

                    if pos >= 15:
                        freq_count_losses[29] = freq_count_losses[29] + 1

                    elif pos <= -15:
                        freq_count_losses[0] = freq_count_losses[0] + 1

                    elif abs(pos) <= 14:
                        freq_count_losses[pos+14] = freq_count_losses[pos+14] + 1

                elif test_games_all[i,4]-test_games_all[i,5] > test_games_all[i,2] and test_games_all[i,7] < test_games_all[i,2]:

                    pos = -1*pos

                    if pos >= 15:
                        freq_count_losses[29] = freq_count_losses[29] + 1

                    elif pos <= -15:
                        freq_count_losses[0] = freq_count_losses[0] + 1

                    elif abs(pos) <= 14:
                        freq_count_losses[pos+14] = freq_count_losses[pos+14] + 1


    labels = np.zeros((TeamList.shape[0],),dtype = object)

    labels[0] = '-15'
    labels[29] = '15'

    for i in range(28):
        labels[i+1] = str(i-14)

    x = np.arange(len(labels))

    width = 0.35  
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, freq_count_wins, width, label='Wins',color = 'lawngreen')
    rects2 = ax.bar(x + width/2, freq_count_losses, width, label='Losses',color = 'tomato')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Vegas Spread')
    ax.set_title('Model ATS')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()




    fig.tight_layout()
    plt.show()

    return










