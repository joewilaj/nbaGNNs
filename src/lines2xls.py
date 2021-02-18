import numpy as np
import scipy.io
import pandas as pd
import warnings
import pdb
import requests
import datetime
import re
import sportsipy
import sys
import pickle

def main():

    year = 2021


    TeamLists = pd.read_excel('data/TeamLists.xls',sheet_name = 0,header = None)
    TeamLists = TeamLists.to_numpy(dtype = object,copy = True)

    TeamList = TeamLists[:,2]
    TeamList_Lines = TeamLists[:,3]

    with open('pickles/NBA_Data_pickled/'+str(year)+'NBAData.pkl', 'rb') as Data: 
        Data_Full = pickle.load(Data)

    Lines = np.zeros((3000,5),dtype = object)

    gamecount = 0

    now = datetime.datetime.now()


    for i in range(30):
        for j in range(364):
            if Data_Full[j,6,i] == 'Home' and (datetime.datetime(year-1,10,12) + datetime.timedelta(days=j)) <= now:


                for m in range(30):
                    if Data_Full[j,7,i] == TeamList[m]:
                        Lines[gamecount,3] = TeamList_Lines[m]
                        break

                Lines[gamecount,2] = TeamList_Lines[i]
                Lines[gamecount,1] = 'home'
                Lines[gamecount,0] = (datetime.datetime(year-1,10,12) + datetime.timedelta(days=j)).date()
                gamecount = gamecount + 1

    Lines = Lines[~np.all(Lines == 0, axis=1)]
    df = pd.DataFrame(Lines)

    datestring = now.strftime("%m_%d_%Y")

    df.to_excel('data/2021Lines_' + datestring + '.xls', index=False, header = False)



if __name__ == "__main__":
    main()

