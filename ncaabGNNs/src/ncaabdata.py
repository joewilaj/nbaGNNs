
import numpy as np
import scipy.io
import pandas as pd
import warnings
import pdb
import requests
import datetime
import re
import sys
import pickle


from sportsipy_copy.sportsipy_me.ncaab.schedule import Schedule
from sportsipy_copy.sportsipy_me.ncaab.teams import Teams


from datetime import date,timedelta


np.set_printoptions(threshold=sys.maxsize)


#Script to pull current game data from  https://www.basketball-reference.com/   via sportsipy: 
#https://github.com/roclark/sportsipy



def main():

    year = 2021

    TeamList = pd.read_excel('data/TeamLists.xls',sheet_name = year-2015,header = None)
    TeamList = TeamList.to_numpy(dtype = object,copy = True)


    now = datetime.datetime.now()
    datestring = now.strftime("%m_%d_%Y")

    today = (now-datetime.datetime(year-1,10,12)).days

    with open('pickles/ncaabdata_pickled/'+str(year)+'ncaabdata.pkl', 'rb') as Data: 
        Data_Full = pickle.load(Data)

    teams = Teams(year)

    for team in teams:
        print(team.name)
        p = 0
        for i in range(TeamList.shape[0]):
            if team.name == TeamList[i]:
                p = p + 1
                break


        if p != 1:
            continue

        schedule = team.schedule
        for game in schedule:

            game_date = game.datetime
            daynum = (game_date - datetime.datetime(year-1,10,12)).days

            if daynum >= today - 5:

                box = game.boxscore


                Data_Full[daynum,7,i] = game.opponent_name
                Data_Full[daynum,8,i] = game.points_for
                Data_Full[daynum,9,i] = game.points_against
                Data_Full[daynum,0,i] = game.game
                Data_Full[daynum,1,i] = game.type


                if game.location == 'Neutral':
                    if min(TeamList[i],game.opponent_name) == TeamList[i]:
                        Data_Full[daynum,6,i] = 'Home'

                    elif min(TeamList[i],game.opponent_name) == game.opponent_name:
                        Data_Full[daynum,6,i] = 'Away'

                else:
                    Data_Full[daynum,6,i] = game.location



                if Data_Full[daynum,6,i] == 'Home':
                    Data_Full[daynum,10,i] = box.home_offensive_rating
                    Data_Full[daynum,11,i] = box.home_defensive_rating
                    Data_Full[daynum,12,i] = box.home_effective_field_goal_percentage
                    Data_Full[daynum,13,i] = box.home_assist_percentage
                    Data_Full[daynum,14,i] = box.home_assists
                    Data_Full[daynum,15,i] = box.home_block_percentage
                    Data_Full[daynum,17,i] = box.home_defensive_rebound_percentage
                    Data_Full[daynum,18,i] = box.home_field_goal_percentage
                    Data_Full[daynum,19,i] = box.home_field_goals
                    Data_Full[daynum,20,i] = box.home_free_throw_attempt_rate
                    Data_Full[daynum,21,i] = box.home_offensive_rebound_percentage
                    Data_Full[daynum,22,i] = box.home_steal_percentage
                    Data_Full[daynum,23,i] = box.home_three_point_field_goal_percentage
                    Data_Full[daynum,24,i] = box.home_turnover_percentage
                    Data_Full[daynum,26,i] = box.home_true_shooting_percentage
                    Data_Full[daynum,27,i] = box.home_free_throws
                    Data_Full[daynum,28,i] = box.pace


                elif Data_Full[daynum,6,i] == 'Away':

                    Data_Full[daynum,10,i] = box.away_offensive_rating
                    Data_Full[daynum,11,i] = box.away_defensive_rating
                    Data_Full[daynum,12,i] = box.away_effective_field_goal_percentage
                    Data_Full[daynum,13,i] = box.away_assist_percentage
                    Data_Full[daynum,14,i] = box.away_assists
                    Data_Full[daynum,15,i] = box.away_block_percentage
                    Data_Full[daynum,17,i] = box.away_defensive_rebound_percentage
                    Data_Full[daynum,18,i] = box.away_field_goal_percentage
                    Data_Full[daynum,19,i] = box.away_field_goals
                    Data_Full[daynum,20,i] = box.away_free_throw_attempt_rate
                    Data_Full[daynum,21,i] = box.away_offensive_rebound_percentage
                    Data_Full[daynum,22,i] = box.away_steal_percentage
                    Data_Full[daynum,23,i] = box.away_three_point_field_goal_percentage
                    Data_Full[daynum,24,i] = box.away_turnover_percentage
                    Data_Full[daynum,26,i] = box.away_true_shooting_percentage
                    Data_Full[daynum,27,i] = box.away_free_throws
                    Data_Full[daynum,28,i] = box.pace


    with open('pickles/ncaabdata_pickled/' + str(year) + 'ncaabdata.pkl', 'wb') as Data_in:  
        pickle.dump(Data_Full,Data_in)
    
    pdb.set_trace()
    


if __name__ == "__main__":
    main()

