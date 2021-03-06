# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import pandas as pd
import pickle
import sys
import pdb
import utils_data



#Construct Statistic Matrices 

def construct_S_orc(Data_Full,schedule,HomeAway,weights,stop):

    N = Data_Full.shape[2]

    Data_Cleared = np.zeros((364,36,N),dtype = object) 

    for k in range(N):
        for r in range(stop):
            Data_Cleared[r,:,k] = Data_Full[r,:,k]
    for k in range(N):
        for r in range(stop):
            Data_Cleared[r,:,k] = Data_Full[r,:,k]



    TPAPROB = np.zeros((N,364),dtype = float)
    TSPROB = np.zeros((N,364),dtype = float)
    AstPROB = np.zeros((N,364),dtype = float)
    STLPROB = np.zeros((N,364),dtype = float)
    BLKPROB = np.zeros((N,364),dtype = float)
    DRBPROB = np.zeros((N,364),dtype = float)
    eFGDefPROB = np.zeros((N,364),dtype = float)
    AstDefPROB = np.zeros((N,364),dtype = float)



    PtsPROB = np.zeros((N,364),dtype = float)
    eFGPROB = np.zeros((N,364),dtype = float)
    OrbPROB = np.zeros((N,364),dtype = float)
    TOPROB = np.zeros((N,364),dtype = float)
    TOoffPROB = np.zeros((N,364),dtype = float)
    FTPROB = np.zeros((N,364),dtype = float)
    P100PROB = np.zeros((N,364),dtype = float)
    PacePROB = np.zeros((N,364),dtype = float)
    FTAPROB = np.zeros((N,364),dtype = float)

    alphaHome = weights[0,17];
    alphaAway = weights[0,18];

    #Construct G_orc


    for t in range(N):

        #Pts Allowed

        PtsVec = np.zeros((364,),dtype = float)
        PtsVecHome = np.zeros((364,),dtype = float)
        PtsVecAway = np.zeros((364,),dtype = float)

            
        for d in range(364):
            if schedule[t,d] != -1:
                ptsd = Data_Cleared[d,8,t]-Data_Cleared[d,9,t]
                if ptsd >= 0:
                    PtsVec[d] = 0
                elif ptsd < 0:
                    PtsVec[d] = -1*ptsd




        for d in range(364):
            if HomeAway[t,d] == 1:
                PtsVecAway[d] = PtsVec[d]
            elif HomeAway[t,d] == 2:
                PtsVecHome[d] = PtsVec[d]

        if np.sum(PtsVecAway) != 0:
            PtsVecAway = (1/np.sum(PtsVecAway))*PtsVecAway

        if np.sum(PtsVecHome) != 0:
            PtsVecHome = (1/np.sum(PtsVecHome))*PtsVecHome



        if np.sum(PtsVecHome) != 0 and np.sum(PtsVecAway) != 0:
            PtsVecFinal = alphaHome*PtsVecHome + alphaAway*PtsVecAway
        elif np.sum(PtsVecHome) == 0 and np.sum(PtsVecAway) != 0:
            PtsVecFinal = PtsVecAway
        elif sum(PtsVecAway) == 0 and np.sum(PtsVecHome) != 0:
            PtsVecFinal = PtsVecHome
        else:
            PtsVecFinal = np.zeros((364,),dtype = float)



        PtsPROB[t,:] = PtsVecFinal

        #PassYardsAllowed


        eFGVec = np.zeros((364,),dtype = float)
        eFGVecHome = np.zeros((364,),dtype = float)
        eFGVecAway = np.zeros((364,),dtype = float)


        for d in range(364):
            if schedule[t,d] != -1:
                opp = schedule[t,d]
                eFGVec[d] = Data_Cleared[d,12,opp]

        for d in range(364):
            if HomeAway[t,d] == 1:
                eFGVecAway[d] = eFGVec[d]
            elif HomeAway[t,d] == 2:
                eFGVecHome[d] = eFGVec[d]

        if np.sum(eFGVecAway) != 0:
            eFGVecAway = (1/np.sum(eFGVecAway))*eFGVecAway

        if np.sum(eFGVecHome) != 0:
            eFGVecHome = (1/np.sum(eFGVecHome))*eFGVecHome


        if np.sum(eFGVecHome) != 0 and np.sum(eFGVecAway) != 0:
            eFGVecFinal = alphaHome*eFGVecHome + alphaAway*eFGVecAway
        elif np.sum(eFGVecHome) == 0 and np.sum(eFGVecAway) != 0:
            eFGVecFinal = eFGVecAway
        elif sum(eFGVecAway) == 0 and np.sum(eFGVecHome) != 0:
            eFGVecFinal = eFGVecHome
        else:
            eFGVecFinal = np.zeros((364,),dtype = float)


        eFGPROB[t,:] = eFGVecFinal

        #Rush Yards Allowed
        
        OrbVec = np.zeros((364,),dtype = float)
        OrbVecHome = np.zeros((364,),dtype = float)
        OrbVecAway = np.zeros((364,),dtype = float)


        for d in range(364):
            if schedule[t,d] != -1:
                OrbVec[d] = Data_Cleared[d,21,opp]

        for d in range(364):
            if HomeAway[t,d] == 1:
                OrbVecAway[d] = OrbVec[d]
            elif HomeAway[t,d] == 2:
                OrbVecHome[d] = OrbVec[d]

        if np.sum(OrbVecAway) != 0:
            OrbVecAway = (1/np.sum(OrbVecAway))*OrbVecAway

        if np.sum(OrbVecHome) != 0:
            OrbVecHome = (1/np.sum(OrbVecHome))*OrbVecHome


        if np.sum(OrbVecHome) != 0 and np.sum(OrbVecAway) != 0:
            OrbVecFinal = alphaHome*OrbVecHome + alphaAway*OrbVecAway
        elif np.sum(OrbVecHome) == 0 and np.sum(OrbVecAway) != 0:
            OrbVecFinal = OrbVecAway
        elif sum(OrbVecAway) == 0 and np.sum(OrbVecHome) != 0:
            OrbVecFinal = OrbVecHome
        else:
            OrbVecFinal = np.zeros((364,),dtype = float)


        OrbPROB[t,:] = OrbVecFinal


        TOVec = np.zeros((364,),dtype = float)
        TOVecHome = np.zeros((364,),dtype = float)
        TOVecAway = np.zeros((364,),dtype = float)


        for d in range(364):
                TOVec[d] = Data_Cleared[d,24,t]/100

        for d in range(364):
            if HomeAway[t,d] == 1:
                TOVecAway[d] = TOVec[d]
            elif HomeAway[t,d] == 2:
                TOVecHome[d] = TOVec[d]

        if np.sum(TOVecAway) != 0:
            TOVecAway = (1/np.sum(TOVecAway))*TOVecAway

        if np.sum(TOVecHome) != 0:
            TOVecHome = (1/np.sum(TOVecHome))*TOVecHome


        if np.sum(TOVecHome) != 0 and np.sum(TOVecAway) != 0:
            TOVecFinal = alphaHome*TOVecHome + alphaAway*TOVecAway
        elif np.sum(TOVecHome) == 0 and np.sum(TOVecAway) != 0:
            TOVecFinal = TOVecAway
        elif sum(TOVecAway) == 0 and np.sum(TOVecHome) != 0:
            TOVecFinal = TOVecHome
        else:
            TOVecFinal = np.zeros((364,),dtype = float)


        TOPROB[t,:] = TOVecFinal


        FTVec = np.zeros((364,),dtype = float)
        FTVecHome = np.zeros((364,),dtype = float)
        FTVecAway = np.zeros((364,),dtype = float)


        for d in range(364):
            if schedule[t,d] != -1:
                FTVec[d] = Data_Cleared[d,27,opp]

        for d in range(364):
            if HomeAway[t,d] == 1:
                FTVecAway[d] = FTVec[d]
            elif HomeAway[t,d] == 2:
                FTVecHome[d] = FTVec[d]

        if np.sum(FTVecAway) != 0:
            FTVecAway = (1/np.sum(FTVecAway))*FTVecAway

        if np.sum(FTVecHome) != 0:
            FTVecHome = (1/np.sum(FTVecHome))*FTVecHome


        if np.sum(FTVecHome) != 0 and np.sum(FTVecAway) != 0:
            FTVecFinal = alphaHome*FTVecHome + alphaAway*FTVecAway
        elif np.sum(FTVecHome) == 0 and np.sum(FTVecAway) != 0:
            FTVecFinal = FTVecAway
        elif sum(FTVecAway) == 0 and np.sum(FTVecHome) != 0:
            FTVecFinal = FTVecHome
        else:
            FTVecFinal = np.zeros((364,),dtype = float)


        FTPROB[t,:] = FTVecFinal



        P100Vec = np.zeros((364,),dtype = float)
        P100VecHome = np.zeros((364,),dtype = float)
        P100VecAway = np.zeros((364,),dtype = float)


        for d in range(364):
            if schedule[t,d] != -1:
                P100Vec[d] = Data_Cleared[d,10,opp]

        for d in range(364):
            if HomeAway[t,d] == 1:
                P100VecAway[d] = P100Vec[d]
            elif HomeAway[t,d] == 2:
                P100VecHome[d] = P100Vec[d]

        if np.sum(P100VecAway) != 0:
            P100VecAway = (1/np.sum(P100VecAway))*P100VecAway

        if np.sum(P100VecHome) != 0:
            P100VecHome = (1/np.sum(P100VecHome))*P100VecHome


        if np.sum(P100VecHome) != 0 and np.sum(P100VecAway) != 0:
            P100VecFinal = alphaHome*P100VecHome + alphaAway*P100VecAway
        elif np.sum(P100VecHome) == 0 and np.sum(P100VecAway) != 0:
            P100VecFinal = P100VecAway
        elif sum(P100VecAway) == 0 and np.sum(P100VecHome) != 0:
            P100VecFinal = P100VecHome
        else:
            P100VecFinal = np.zeros((364,),dtype = float)


        P100PROB[t,:] = P100VecFinal



        TOoffVec = np.zeros((364,),dtype = float)
        TOoffVecHome = np.zeros((364,),dtype = float)
        TOoffVecAway = np.zeros((364,),dtype = float)


        for d in range(364):
            if schedule[t,d] != -1:
                opp = schedule[t,d]
                TOoffVec[d] = 1 - (Data_Cleared[d,24,opp]/100)

        for d in range(364):
            if HomeAway[t,d] == 1:
                TOoffVecAway[d] = TOoffVec[d]
            elif HomeAway[t,d] == 2:
                TOoffVecHome[d] = TOoffVec[d]

        if np.sum(TOoffVecAway) != 0:
            TOoffVecAway = (1/np.sum(TOoffVecAway))*TOoffVecAway

        if np.sum(TOoffVecHome) != 0:
            TOoffVecHome = (1/np.sum(TOoffVecHome))*TOoffVecHome


        if np.sum(TOoffVecHome) != 0 and np.sum(TOoffVecAway) != 0:
            TOoffVecFinal = alphaHome*TOoffVecHome + alphaAway*TOoffVecAway
        elif np.sum(TOoffVecHome) == 0 and np.sum(TOoffVecAway) != 0:
            TOoffVecFinal = TOoffVecAway
        elif sum(TOoffVecAway) == 0 and np.sum(TOoffVecHome) != 0:
            TOoffVecFinal = TOoffVecHome
        else:
            TOoffVecFinal = np.zeros((364,),dtype = float)


        TOoffPROB[t,:] = TOoffVecFinal

        #Pass Yards Per Attempt Allowed 

        FTAVec = np.zeros((364,),dtype = float)
        FTAVecHome = np.zeros((364,),dtype = float)
        FTAVecAway = np.zeros((364,),dtype = float)


        for d in range(364):
            if schedule[t,d] != -1:
                opp = schedule[t,d]
                FTAVec[d] = Data_Cleared[d,20,opp]

        for d in range(364):
            if HomeAway[t,d] == 1:
                FTAVecAway[d] = FTAVec[d]
            elif HomeAway[t,d] == 2:
                FTAVecHome[d] = FTAVec[d]

        if np.sum(FTAVecAway) != 0:
            FTAVecAway = (1/np.sum(FTAVecAway))*FTAVecAway

        if np.sum(FTAVecHome) != 0:
            FTAVecHome = (1/np.sum(FTAVecHome))*FTAVecHome


        if np.sum(FTAVecHome) != 0 and np.sum(FTAVecAway) != 0:
            FTAVecFinal = alphaHome*FTAVecHome + alphaAway*FTAVecAway
        elif np.sum(FTAVecHome) == 0 and np.sum(FTAVecAway) != 0:
            FTAVecFinal = FTAVecAway
        elif sum(FTAVecAway) == 0 and np.sum(FTAVecHome) != 0:
            FTAVecFinal = FTAVecHome
        else:
            FTAVecFinal = np.zeros((364,),dtype = float)


        FTAPROB[t,:] = FTAVecFinal



        TPAVec = np.zeros((364,),dtype = float)
        TPAVecHome = np.zeros((364,),dtype = float)
        TPAVecAway = np.zeros((364,),dtype = float)


        for d in range(364):
            if schedule[t,d] != -1:
                opp = schedule[t,d]
                TPAVec[d] = Data_Cleared[d,23,opp]

        for d in range(364):
            if HomeAway[t,d] == 1:
                TPAVecAway[d] = TPAVec[d]
            elif HomeAway[t,d] == 2:
                TPAVecHome[d] = TPAVec[d]

        if np.sum(TPAVecAway) != 0:
            TPAVecAway = (1/np.sum(TPAVecAway))*TPAVecAway

        if np.sum(TPAVecHome) != 0:
            TPAVecHome = (1/np.sum(TPAVecHome))*TPAVecHome


        if np.sum(TPAVecHome) != 0 and np.sum(TPAVecAway) != 0:
            TPAVecFinal = alphaHome*TPAVecHome + alphaAway*TPAVecAway
        elif np.sum(TPAVecHome) == 0 and np.sum(TPAVecAway) != 0:
            TPAVecFinal = TPAVecAway
        elif sum(TPAVecAway) == 0 and np.sum(TPAVecHome) != 0:
            TPAVecFinal = TPAVecHome
        else:
            TPAVecFinal = np.zeros((364,),dtype = float)


        TPAPROB[t,:] = TPAVecFinal

        #Sacks Allowed


        TSVec = np.zeros((364,),dtype = float)
        TSVecHome = np.zeros((364,),dtype = float)
        TSVecAway = np.zeros((364,),dtype = float)


        for d in range(364):
            if schedule[t,d] != -1:
                opp = schedule[t,d]
                TSVec[d] = Data_Cleared[d,26,opp]

        for d in range(364):
            if HomeAway[t,d] == 1:
                TSVecAway[d] = TSVec[d]
            elif HomeAway[t,d] == 2:
                TSVecHome[d] = TSVec[d]

        if np.sum(TSVecAway) != 0:
            TSVecAway = (1/np.sum(TSVecAway))*TSVecAway

        if np.sum(TSVecHome) != 0:
            TSVecHome = (1/np.sum(TSVecHome))*TSVecHome


        if np.sum(TSVecHome) != 0 and np.sum(TSVecAway) != 0:
            TSVecFinal = alphaHome*TSVecHome + alphaAway*TSVecAway
        elif np.sum(TSVecHome) == 0 and np.sum(TSVecAway) != 0:
            TSVecFinal = TSVecAway
        elif sum(TSVecAway) == 0 and np.sum(TSVecHome) != 0:
            TSVecFinal = TSVecHome
        else:
            TSVecFinal = np.zeros((364,),dtype = float)


        TSPROB[t,:] = TSVecFinal


        AstVec = np.zeros((364,),dtype = float)
        AstVecHome = np.zeros((364,),dtype = float)
        AstVecAway = np.zeros((364,),dtype = float)


        for d in range(364):
            if schedule[t,d] != -1:
                opp = schedule[t,d]
                AstVec[d] = Data_Cleared[d,13,opp]

        for d in range(364):
            if HomeAway[t,d] == 1:
                AstVecAway[d] = AstVec[d]
            elif HomeAway[t,d] == 2:
                AstVecHome[d] = AstVec[d]

        if np.sum(AstVecAway) != 0:
            AstVecAway = (1/np.sum(AstVecAway))*AstVecAway

        if np.sum(AstVecHome) != 0:
            AstVecHome = (1/np.sum(AstVecHome))*AstVecHome


        if np.sum(AstVecHome) != 0 and np.sum(AstVecAway) != 0:
            AstVecFinal = alphaHome*AstVecHome + alphaAway*AstVecAway
        elif np.sum(AstVecHome) == 0 and np.sum(AstVecAway) != 0:
            AstVecFinal = AstVecAway
        elif sum(AstVecAway) == 0 and np.sum(AstVecHome) != 0:
            AstVecFinal = AstVecHome
        else:
            AstVecFinal = np.zeros((364,),dtype = float)


        AstPROB[t,:] = AstVecFinal


        #STL
        STLVec = np.zeros((364,),dtype = float)

        STLVecHome = np.zeros((364,),dtype = float)
        STLVecAway = np.zeros((364,),dtype = float)

        for d in range(364):
            if schedule[t,d] != -1:
                opp = schedule[t,d]
                STLVec[d] = Data_Cleared[d,22,opp]




        for d in range(364):
            if HomeAway[t,d] == 1:
                STLVecAway[d] = STLVec[d]
            elif HomeAway[t,d] == 2:
                STLVecHome[d] = STLVec[d]
  

        if np.sum(STLVecAway) != 0:
            STLVecAway = (1/sum(STLVecAway))*STLVecAway



        if np.sum(STLVecHome) != 0:
            STLVecHome = (1/np.sum(STLVecHome))*STLVecHome




        if np.sum(STLVecHome) != 0 and np.sum(STLVecAway) != 0:
            STLVecFinal = alphaHome*STLVecHome + alphaAway*STLVecAway
        elif np.sum(STLVecHome) == 0 and np.sum(STLVecAway) != 0:
            STLVecFinal = STLVecAway
        elif np.sum(STLVecAway) == 0 and sum(STLVecHome) != 0:
            STLVecFinal = STLVecHome
        else:
            STLVecFinal = np.zeros((364,),dtype = float)



        STLPROB[t,:] = STLVecFinal

        #BLK

        BLKVec = np.zeros((364,),dtype = float)
        BLKVecHome = np.zeros((364,),dtype = float)
        BLKVecAway = np.zeros((364,),dtype = float)


        for d in range(364):
            if schedule[t,d] != -1:
                opp = schedule[t,d]
                BLKVec[d] = Data_Cleared[d,15,opp]

        for d in range(364):
            if HomeAway[t,d] == 1:
                BLKVecAway[d] = BLKVec[d]
            elif HomeAway[t,d] == 2:
                BLKVecHome[d] = BLKVec[d]

        if np.sum(BLKVecAway) != 0:
            BLKVecAway = (1/np.sum(BLKVecAway))*BLKVecAway

        if np.sum(BLKVecHome) != 0:
            BLKVecHome = (1/np.sum(BLKVecHome))*BLKVecHome


        if np.sum(BLKVecHome) != 0 and np.sum(BLKVecAway) != 0:
            BLKVecFinal = alphaHome*BLKVecHome + alphaAway*BLKVecAway
        elif np.sum(BLKVecHome) == 0 and np.sum(BLKVecAway) != 0:
            BLKVecFinal = BLKVecAway
        elif sum(BLKVecAway) == 0 and np.sum(BLKVecHome) != 0:
            BLKVecFinal = BLKVecHome
        else:
            BLKVecFinal = np.zeros((364,),dtype = float)


        BLKPROB[t,:] = BLKVecFinal



        DRBVec = np.zeros((364,),dtype = float)
        DRBVecHome = np.zeros((364,),dtype = float)
        DRBVecAway = np.zeros((364,),dtype = float)


        for d in range(364):
            if schedule[t,d] != -1:
                DRBVec[d] = 100 - Data_Cleared[d,21,t]

        for d in range(364):
            if HomeAway[t,d] == 1:
                DRBVecAway[d] = DRBVec[d]
            elif HomeAway[t,d] == 2:
                DRBVecHome[d] = DRBVec[d]

        if np.sum(DRBVecAway) != 0:
            DRBVecAway = (1/np.sum(DRBVecAway))*DRBVecAway

        if np.sum(DRBVecHome) != 0:
            DRBVecHome = (1/np.sum(DRBVecHome))*DRBVecHome


        if np.sum(DRBVecHome) != 0 and np.sum(DRBVecAway) != 0:
            DRBVecFinal = alphaHome*DRBVecHome + alphaAway*DRBVecAway
        elif np.sum(DRBVecHome) == 0 and np.sum(DRBVecAway) != 0:
            DRBVecFinal = DRBVecAway
        elif sum(DRBVecAway) == 0 and np.sum(DRBVecHome) != 0:
            DRBVecFinal = DRBVecHome
        else:
            DRBVecFinal = np.zeros((364,),dtype = float)


        DRBPROB[t,:] = DRBVecFinal

        #Fourth Down Coversions

        eFGDefVec = np.zeros((364,),dtype = float)
        eFGDefVecHome = np.zeros((364,),dtype = float)
        eFGDefVecAway = np.zeros((364,),dtype = float)


        for d in range(364):
            if schedule[t,d] != -1:
                eFGDefVec[d] = 1 - Data_Cleared[d,12,t]


        for d in range(364):
            if HomeAway[t,d] == 1:
                eFGDefVecAway[d] = eFGDefVec[d]
            elif HomeAway[t,d] == 2:
                eFGDefVecHome[d] = eFGDefVec[d]

        if np.sum(eFGDefVecAway) != 0:
            eFGDefVecAway = (1/np.sum(eFGDefVecAway))*eFGDefVecAway

        if np.sum(eFGDefVecHome) != 0:
            eFGDefVecHome = (1/np.sum(eFGDefVecHome))*eFGDefVecHome


        if np.sum(eFGDefVecHome) != 0 and np.sum(eFGDefVecAway) != 0:
            eFGDefVecFinal = alphaHome*eFGDefVecHome + alphaAway*eFGDefVecAway
        elif np.sum(eFGDefVecHome) == 0 and np.sum(eFGDefVecAway) != 0:
            eFGDefVecFinal = eFGDefVecAway
        elif sum(eFGDefVecAway) == 0 and np.sum(eFGDefVecHome) != 0:
            eFGDefVecFinal = eFGDefVecHome
        else:
            eFGDefVecFinal = np.zeros((364,),dtype = float)


        eFGDefPROB[t,:] = eFGDefVecFinal


        AstDefVec = np.zeros((364,),dtype = float)
        AstDefVecHome = np.zeros((364,),dtype = float)
        AstDefVecAway = np.zeros((364,),dtype = float)


        for d in range(364):
            if schedule[t,d] != -1:
                AstDefVec[d] = 100 - Data_Cleared[d,13,t]

        if np.sum(AstDefVec) != 0:
            AstDefVecFinal = (1/np.sum(AstDefVec))*AstDefVec  
        else:
            AstDefVecFinal = np.zeros((364,),dtype = float)

        AstDefPROB[t,:] = AstDefVecFinal







    TPA = np.zeros((N,N),dtype = float)
    TS = np.zeros((N,N),dtype = float)
    Ast = np.zeros((N,N),dtype = float)
    STL = np.zeros((N,N),dtype = float)
    BLK = np.zeros((N,N),dtype = float)
    DRB = np.zeros((N,N),dtype = float)
    eFGDef = np.zeros((N,N),dtype = float)
    AstDef = np.zeros((N,N),dtype = float)


    for k in range(N):
        rowfinal = np.zeros((N,),dtype = float)
        for j in range(364):
            g = schedule[k,j]
            if g != -1:
                rowfinal[g] = rowfinal[g] + TPAPROB[k,j]  
        TPA[k,:] = rowfinal
        if np.sum(TPA[k,:]) != 0:
            TPA[k,:] = (1/np.sum(TPA[k,:]))*TPA[k,:]
        elif np.sum(TPA[k,:]) == 0:
            TPA[k,:] = np.ones(((1,N)),dtype = float)



    for k in range(N):
        rowfinal = np.zeros((N,),dtype = float)
        for j in range(364):
            g = schedule[k,j]
            if g != -1:
                rowfinal[g] = rowfinal[g] + TSPROB[k,j]  
        TS[k,:] = rowfinal
        if np.sum(TS[k,:]) != 0:
            TS[k,:] = (1/np.sum(TS[k,:]))*TS[k,:]


    for k in range(N):
        rowfinal = np.zeros((N,),dtype = float)
        for j in range(364):
            g = schedule[k,j]
            if g != -1:
                rowfinal[g] = rowfinal[g] + AstPROB[k,j]  
        Ast[k,:] = rowfinal
        if np.sum(Ast[k,:]) != 0:
            Ast[k,:] = (1/np.sum(Ast[k,:]))*Ast[k,:]


    for k in range(N):
        rowfinal = np.zeros((N,),dtype = float)
        for j in range(364):
            g = schedule[k,j]
            if g != -1:
                rowfinal[g] = rowfinal[g] + STLPROB[k,j]  
        STL[k,:] = rowfinal
        if np.sum(STL[k,:]) != 0:
            STL[k,:] = (1/np.sum(STL[k,:]))*STL[k,:]


    for k in range(N):
        rowfinal = np.zeros((N,),dtype = float)
        for j in range(364):
            g = schedule[k,j]
            if g != -1:
                rowfinal[g] = rowfinal[g] + BLKPROB[k,j]  
        BLK[k,:] = rowfinal
        if np.sum(BLK[k,:]) != 0:
            BLK[k,:] = (1/np.sum(BLK[k,:]))*BLK[k,:]


    for k in range(N):
        rowfinal = np.zeros((N,),dtype = float)
        for j in range(364):
            g = schedule[k,j]
            if g != -1:
                rowfinal[g] = rowfinal[g] + DRBPROB[k,j]  
        DRB[k,:] = rowfinal
        if np.sum(DRB[k,:]) != 0:
            DRB[k,:] = (1/np.sum(DRB[k,:]))*DRB[k,:]


    for k in range(N):
        rowfinal = np.zeros((N,),dtype = float)
        for j in range(364):
            g = schedule[k,j]
            if g != -1:
                rowfinal[g] = rowfinal[g] + eFGDefPROB[k,j]  
        eFGDef[k,:] = rowfinal
        if np.sum(eFGDef[k,:]) != 0:
            eFGDef[k,:] = (1/np.sum(eFGDef[k,:]))*eFGDef[k,:]


    for k in range(N):
        rowfinal = np.zeros((N,),dtype = float)
        for j in range(364):
            g = schedule[k,j]
            if g != -1:
                rowfinal[g] = rowfinal[g] + AstDefPROB[k,j]  
        AstDef[k,:] = rowfinal
        if np.sum(AstDef[k,:]) != 0:
            AstDef[k,:] = (1/np.sum(AstDef[k,:]))*AstDef[k,:]

    Pts = np.zeros((N,N),dtype = float)
    eFG = np.zeros((N,N),dtype = float)
    Orb = np.zeros((N,N),dtype = float)
    TO = np.zeros((N,N),dtype = float)
    FT = np.zeros((N,N),dtype = float)
    P100 = np.zeros((N,N),dtype = float)
    TOoff = np.zeros((N,N),dtype = float)
    FTA = np.zeros((N,N),dtype = float)


    for k in range(N):
        rowfinal = np.zeros((N,),dtype = float)
        for j in range(364):
            g = schedule[k,j]
            if g != -1:
                rowfinal[g] = rowfinal[g] + PtsPROB[k,j]  
        Pts[k,:] = rowfinal
        if np.sum(Pts[k,:]) != 0:
            Pts[k,:] = (1/np.sum(Pts[k,:]))*Pts[k,:]


    for k in range(N):
        rowfinal = np.zeros((N,),dtype = float)
        for j in range(364):
            g = schedule[k,j]
            if g != -1:
                rowfinal[g] = rowfinal[g] + eFGPROB[k,j]  
        eFG[k,:] = rowfinal
        if np.sum(eFG[k,:]) != 0:
            eFG[k,:] = (1/np.sum(eFG[k,:]))*eFG[k,:]


    for k in range(N):
        rowfinal = np.zeros((N,),dtype = float)
        for j in range(364):
            g = schedule[k,j]
            if g != -1:
                rowfinal[g] = rowfinal[g] + OrbPROB[k,j]  
        Orb[k,:] = rowfinal
        if np.sum(Orb[k,:]) != 0:
            Orb[k,:] = (1/np.sum(Orb[k,:]))*Orb[k,:]


    for k in range(N):
        rowfinal = np.zeros((N,),dtype = float)
        for j in range(364):
            g = schedule[k,j]
            if g != -1:
                rowfinal[g] = rowfinal[g] + TOPROB[k,j]  
        TO[k,:] = rowfinal
        if np.sum(TO[k,:]) != 0:
            TO[k,:] = (1/np.sum(TO[k,:]))*TO[k,:]


    for k in range(N):
        rowfinal = np.zeros((N,),dtype = float)
        for j in range(364):
            g = schedule[k,j]
            if g != -1:
                rowfinal[g] = rowfinal[g] + FTPROB[k,j]  
        FT[k,:] = rowfinal
        if np.sum(FT[k,:]) != 0:
            FT[k,:] = (1/np.sum(FT[k,:]))*FT[k,:]


    for k in range(N):
        rowfinal = np.zeros((N,),dtype = float)
        for j in range(364):
            g = schedule[k,j]
            if g != -1:
                rowfinal[g] = rowfinal[g] + P100PROB[k,j]  
        P100[k,:] = rowfinal
        if np.sum(P100[k,:]) != 0:
            P100[k,:] = (1/np.sum(P100[k,:]))*P100[k,:]



    for k in range(N):
        rowfinal = np.zeros((N,),dtype = float)
        for j in range(364):
            g = schedule[k,j]
            if g != -1:
                rowfinal[g] = rowfinal[g] + TOoffPROB[k,j]  
        TOoff[k,:] = rowfinal
        if np.sum(TOoff[k,:]) != 0:
            TOoff[k,:] = (1/np.sum(TOoff[k,:]))*TOoff[k,:]



    for k in range(N):
        rowfinal = np.zeros((N,),dtype = float)
        for j in range(364):
            g = schedule[k,j]
            if g != -1:
                rowfinal[g] = rowfinal[g] + FTAPROB[k,j]  
        FTA[k,:] = rowfinal
        if np.sum(FTA[k,:]) != 0:
            FTA[k,:] = (1/np.sum(FTA[k,:]))*FTA[k,:]



    for i in range(N):
        for j in range(N):
            gc = 0
            for k in range(stop):
                if schedule[i,k] == j:
                    gc = gc + 1

            if gc != 0:
                Pts[i,j] = Pts[i,j]/gc
                eFG[i,j] = eFG[i,j]/gc
                Orb[i,j] = Orb[i,j]/gc
                TO[i,j] = TO[i,j]/gc
                FT[i,j] = FT[i,j]/gc
                P100[i,j] = P100[i,j]/gc
                TOoff[i,j] = TOoff[i,j]/gc
                FTA[i,j] = FTA[i,j]/gc
                TPA[i,j] = TPA[i,j]/gc
                TS[i,j] = TS[i,j]/gc
                Ast[i,j] = Ast[i,j]/gc
                STL[i,j] = STL[i,j]/gc
                BLK[i,j] = BLK[i,j]/gc
                DRB[i,j] = FTA[i,j]/gc
                eFGDef[i,j] = eFGDef[i,j]/gc
                AstDef[i,j] = AstDef[i,j]/gc


    #Add Oracle Adjustment



    TPA = np.concatenate((TPA,np.zeros((1,N),dtype = float)),axis = 0)
    TPA = np.concatenate((TPA,np.zeros((N+1,1),dtype = float)),axis = 1)

    TS = np.concatenate((TS,np.zeros((1,N),dtype = float)),axis = 0)
    TS = np.concatenate((TS,np.zeros((N+1,1),dtype = float)),axis = 1)

    Ast = np.concatenate((Ast,np.zeros((1,N),dtype = float)),axis = 0)
    Ast = np.concatenate((Ast,np.zeros((N+1,1),dtype = float)),axis = 1)

    STL = np.concatenate((STL,np.zeros((1,N),dtype = float)),axis = 0)
    STL = np.concatenate((STL,np.zeros((N+1,1),dtype = float)),axis = 1)


    BLK = np.concatenate((BLK,np.zeros((1,N),dtype = float)),axis = 0)
    BLK = np.concatenate((BLK,np.zeros((N+1,1),dtype = float)),axis = 1)

    DRB = np.concatenate((DRB,np.zeros((1,N),dtype = float)),axis = 0)
    DRB = np.concatenate((DRB,np.zeros((N+1,1),dtype = float)),axis = 1)

    eFGDef = np.concatenate((eFGDef,np.zeros((1,N),dtype = float)),axis = 0)
    eFGDef = np.concatenate((eFGDef,np.zeros((N+1,1),dtype = float)),axis = 1)

    AstDef = np.concatenate((AstDef,np.zeros((1,N),dtype = float)),axis = 0)
    AstDef = np.concatenate((AstDef,np.zeros((N+1,1),dtype = float)),axis = 1)


    for c in range(N+1):
        TPA[N,c] = np.sum(TPA[:,c])


    if np.sum(TPA[N,:]) != 0:
        TPA[N,:] = (1/np.sum(TPA[N,:]))*TPA[N,:]






    for c in range(N+1):
        TS[N,c] = np.sum(TS[:,c])


    if np.sum(TS[N,:]) != 0:
        TS[N,:] = (1/np.sum(TS[N,:]))*TS[N,:]


    for c in range(N+1):
        Ast[N,c] = np.sum(Ast[:,c])


    if np.sum(Ast[N,:]) != 0:
        Ast[N,:] = (1/np.sum(Ast[N,:]))*Ast[N,:]


    for c in range(N+1):
        STL[N,c] = np.sum(STL[:,c])


    if np.sum(STL[N,:]) != 0:
        STL[N,:] = (1/np.sum(STL[N,:]))*STL[N,:]



    for c in range(N+1):
        BLK[N,c] = np.sum(BLK[:,c])


    if np.sum(BLK[N,:]) != 0:
        BLK[N,:] = (1/np.sum(BLK[N,:]))*BLK[N,:]


    for c in range(N+1):
        DRB[N,c] = np.sum(DRB[:,c])


    if np.sum(DRB[N,:]) != 0:
        DRB[N,:] = (1/np.sum(DRB[N,:]))*DRB[N,:]


    for c in range(N+1):
        eFGDef[N,c] = np.sum(eFGDef[:,c])


    if np.sum(eFGDef[N,:]) != 0:
        eFGDef[N,:] = (1/np.sum(eFGDef[N,:]))*eFGDef[N,:]


    for c in range(N+1):
        AstDef[N,c] = np.sum(AstDef[:,c])


    if np.sum(AstDef[N,:]) != 0:
        AstDef[N,:] = (1/np.sum(AstDef[N,:]))*AstDef[N,:]


    #Oracle Columns D

    for r in range(N):
        nonzerocount = 0
        for col in range(N):
            if TPA[r,col] != 0:
                nonzerocount = nonzerocount + 1

        if nonzerocount != 0:
            TPA[r,N] = 1/nonzerocount
            TPA[r,:] = 1/(np.sum(TPA[r,:]))*TPA[r,:]




    for r in range(N):
        nonzerocount = 0
        for col in range(N):
            if TS[r,col] != 0:
                nonzerocount = nonzerocount + 1

        if nonzerocount != 0:
            TS[r,N] = 1/nonzerocount
            TS[r,:] = 1/(np.sum(TS[r,:]))*TS[r,:]


    for r in range(N):
        nonzerocount = 0
        for col in range(N):
            if Ast[r,col] != 0:
                nonzerocount = nonzerocount + 1

        if nonzerocount != 0:
            Ast[r,N] = 1/nonzerocount
            Ast[r,:] = 1/(np.sum(Ast[r,:]))*Ast[r,:]



    for r in range(N):
        nonzerocount = 0
        for col in range(N):
            if STL[r,col] != 0:
                nonzerocount = nonzerocount + 1

        if nonzerocount != 0:
            STL[r,N] = 1/nonzerocount
            STL[r,:] = 1/(np.sum(STL[r,:]))*STL[r,:]


    for r in range(N):
        nonzerocount = 0
        for col in range(N):
            if BLK[r,col] != 0:
                nonzerocount = nonzerocount + 1

        if nonzerocount != 0:
            BLK[r,N] = 1/nonzerocount
            BLK[r,:] = 1/(np.sum(BLK[r,:]))*BLK[r,:]


    for r in range(N):
        nonzerocount = 0
        for col in range(N):
            if DRB[r,col] != 0:
                nonzerocount = nonzerocount + 1

        if nonzerocount != 0:
            DRB[r,N] = 1/nonzerocount
            DRB[r,:] = 1/(np.sum(DRB[r,:]))*DRB[r,:]


    for r in range(N):
        nonzerocount = 0
        for col in range(N):
            if eFGDef[r,col] != 0:
                nonzerocount = nonzerocount + 1

        if nonzerocount != 0:
            eFGDef[r,N] = 1/nonzerocount
            eFGDef[r,:] = 1/(np.sum(eFGDef[r,:]))*eFGDef[r,:]


    for r in range(N):
        nonzerocount = 0
        for col in range(N):
            if AstDef[r,col] != 0:
                nonzerocount = nonzerocount + 1

        if nonzerocount != 0:
            AstDef[r,N] = 1/nonzerocount
            AstDef[r,:] = 1/(np.sum(AstDef[r,:]))*AstDef[r,:]



    Pts = np.concatenate((Pts,np.zeros((1,N),dtype = float)),axis = 0)
    Pts = np.concatenate((Pts,np.zeros((N+1,1),dtype = float)),axis = 1)

    eFG = np.concatenate((eFG,np.zeros((1,N),dtype = float)),axis = 0)
    eFG = np.concatenate((eFG,np.zeros((N+1,1),dtype = float)),axis = 1)

    Orb = np.concatenate((Orb,np.zeros((1,N),dtype = float)),axis = 0)
    Orb = np.concatenate((Orb,np.zeros((N+1,1),dtype = float)),axis = 1)

    TO = np.concatenate((TO,np.zeros((1,N),dtype = float)),axis = 0)
    TO = np.concatenate((TO,np.zeros((N+1,1),dtype = float)),axis = 1)


    FT = np.concatenate((FT,np.zeros((1,N),dtype = float)),axis = 0)
    FT = np.concatenate((FT,np.zeros((N+1,1),dtype = float)),axis = 1)

    P100 = np.concatenate((P100,np.zeros((1,N),dtype = float)),axis = 0)
    P100 = np.concatenate((P100,np.zeros((N+1,1),dtype = float)),axis = 1)

    TOoff = np.concatenate((TOoff,np.zeros((1,N),dtype = float)),axis = 0)
    TOoff = np.concatenate((TOoff,np.zeros((N+1,1),dtype = float)),axis = 1)

    FTA = np.concatenate((FTA,np.zeros((1,N),dtype = float)),axis = 0)
    FTA = np.concatenate((FTA,np.zeros((N+1,1),dtype = float)),axis = 1)


    for c in range(N+1):
        Pts[N,c] = np.sum(Pts[:,c])


    if np.sum(Pts[N,:]) != 0:
        Pts[N,:] = (1/np.sum(Pts[N,:]))*Pts[N,:]



    for c in range(N+1):
        eFG[N,c] = np.sum(eFG[:,c])


    if np.sum(eFG[N,:]) != 0:
        eFG[N,:] = (1/np.sum(eFG[N,:]))*eFG[N,:]


    for c in range(N+1):
        Orb[N,c] = np.sum(Orb[:,c])


    if np.sum(Orb[N,:]) != 0:
        Orb[N,:] = (1/np.sum(Orb[N,:]))*Orb[N,:]


    for c in range(N+1):
        TO[N,c] = np.sum(TO[:,c])


    if np.sum(STL[N,:]) != 0:
        TO[N,:] = (1/np.sum(TO[N,:]))*TO[N,:]



    for c in range(N+1):
        FT[N,c] = np.sum(FT[:,c])


    if np.sum(FT[N,:]) != 0:
        FT[N,:] = (1/np.sum(FT[N,:]))*FT[N,:]


    for c in range(N+1):
        P100[N,c] = np.sum(P100[:,c])


    if np.sum(P100[N,:]) != 0:
        P100[N,:] = (1/np.sum(P100[N,:]))*P100[N,:]


    for c in range(N+1):
        TOoff[N,c] = np.sum(TOoff[:,c])


    if np.sum(TOoff[N,:]) != 0:
        TOoff[N,:] = (1/np.sum(TOoff[N,:]))*TOoff[N,:]


    for c in range(N+1):
        FTA[N,c] = np.sum(FTA[:,c])


    if np.sum(FTA[N,:]) != 0:
        FTA[N,:] = (1/np.sum(FTA[N,:]))*FTA[N,:]


    #Oracle Columns D

    for r in range(N):
        nonzerocount = 0
        for col in range(N):
            if Pts[r,col] != 0:
                nonzerocount = nonzerocount + 1

        if nonzerocount != 0:
            Pts[r,N] = 1/nonzerocount
            Pts[r,:] = 1/(np.sum(Pts[r,:]))*Pts[r,:]


    for r in range(N):
        nonzerocount = 0
        for col in range(N):
            if eFG[r,col] != 0:
                nonzerocount = nonzerocount + 1

        if nonzerocount != 0:
            eFG[r,N] = 1/nonzerocount
            eFG[r,:] = 1/(np.sum(eFG[r,:]))*eFG[r,:]


    for r in range(N):
        nonzerocount = 0
        for col in range(N):
            if Orb[r,col] != 0:
                nonzerocount = nonzerocount + 1

        if nonzerocount != 0:
            Orb[r,N] = 1/nonzerocount
            Orb[r,:] = 1/(np.sum(Orb[r,:]))*Orb[r,:]



    for r in range(N):
        nonzerocount = 0
        for col in range(N):
            if TO[r,col] != 0:
                nonzerocount = nonzerocount + 1

        if nonzerocount != 0:
            TO[r,N] = 1/nonzerocount
            TO[r,:] = 1/(np.sum(TO[r,:]))*TO[r,:]


    for r in range(N):
        nonzerocount = 0
        for col in range(N):
            if FT[r,col] != 0:
                nonzerocount = nonzerocount + 1

        if nonzerocount != 0:
            FT[r,N] = 1/nonzerocount
            FT[r,:] = 1/(np.sum(FT[r,:]))*FT[r,:]


    for r in range(N):
        nonzerocount = 0
        for col in range(N):
            if P100[r,col] != 0:
                nonzerocount = nonzerocount + 1

        if nonzerocount != 0:
            P100[r,N] = 1/nonzerocount
            P100[r,:] = 1/(np.sum(P100[r,:]))*P100[r,:]


    for r in range(N):
        nonzerocount = 0
        for col in range(N):
            if TOoff[r,col] != 0:
                nonzerocount = nonzerocount + 1

        if nonzerocount != 0:
            TOoff[r,N] = 1/nonzerocount
            TOoff[r,:] = 1/(np.sum(TOoff[r,:]))*TOoff[r,:]


    for r in range(N):
        nonzerocount = 0
        for col in range(N):
            if FTA[r,col] != 0:
                nonzerocount = nonzerocount + 1

        if nonzerocount != 0:
            FTA[r,N] = 1/nonzerocount
            FTA[r,:] = 1/(np.sum(FTA[r,:]))*FTA[r,:]





    S_orc = weights[0,0]*Pts + weights[0,1]*eFG + weights[0,2]*Orb +weights[0,3]*FT \
   + weights[0,4]*FTA + weights[0,5]*Ast + weights[0,6]*TPA + weights[0,7]*P100 \
   + weights[0,8]*TOoff + weights[0,9]*TS + weights[0,10]*STL + weights[0,11]*BLK \
   + weights[0,12]*TO + weights[0,13]*DRB + weights[0,14]*eFGDef + weights[0,15]*AstDef

    ObyD = weights[0,11]*STL + weights[0,12]*BLK + weights[0,1]*Pts \
         + weights[0,13]*TO + weights[0,14]*DRB + weights[0,15]*eFGDef + weights[0,16]*AstDef

    DbyO = weights[0,0]*Pts + weights[0,2]*eFG + weights[0,3]*Orb +weights[0,4]*FT \
         + weights[0,5]*FTA + weights[0,6]*Ast + weights[0,8]*P100 \
         + weights[0,9]*TOoff + weights[0,10]*TS


    for i in range(ObyD.shape[0]):
        nzc = np.count_nonzero(ObyD[i,:])

        if nzc != 0:
            mu = np.sum(ObyD[i,:])/nzc

            for j in range(ObyD.shape[1]):
                if ObyD[i,j] != 0:
                    ObyD[i,j] = 1/(1+ np.exp(-1000*(ObyD[i,j]-mu)))


    for i in range(DbyO.shape[0]):
        nzc = np.count_nonzero(DbyO[i,:])

        if nzc != 0:
            mu = np.sum(DbyO[i,:])/nzc

            for j in range(DbyO.shape[1]):
                if DbyO[i,j] != 0:
                    DbyO[i,j] = 1/(1 + np.exp(-1000*(DbyO[i,j]-mu)))



    
    #Concatentate Off and Def statistic matrices with zeros to create SOffDef

    S_orc_ObyD = np.concatenate((np.zeros((N+1,N+1),dtype = float),ObyD),axis = 1)
    S_orc_DbyO = np.concatenate((DbyO,np.zeros((N+1,N+1),dtype = float)),axis = 1)

    S_OffDef = np.concatenate((S_orc_ObyD,S_orc_DbyO),axis = 0)


    for i in range(S_OffDef.shape[0]):
        for j in range(S_OffDef.shape[1]):
            if S_OffDef[i,j] != 0 and S_OffDef[i,j] < 0.01:
                S_OffDef[i,j] = 0



    A_OffDef = np.zeros((2*(N+1),2*(N+1)),dtype = float)

    for m in range(N+1):
        for k in range(N+1):
            if Pts[m,k] != 0:
                A_OffDef[m+N+1,k] = 1
                A_OffDef[m,k+N+1] = 1




    S_OffDef = utils_data.sto_mat(S_OffDef)



    return S_OffDef, A_OffDef


#Compute PageRank Separately on Offense and Defense nodes

def PageRank(G_orc,TeamList):

    N = TeamList.shape[0]


    Pi = (1/2*(N+1))*np.ones((2*(N+1),1),dtype = float)

    DiffNorm = 1

    while DiffNorm >= .00001:
        DiffNorm = 0

        PiNew = np.dot(G_orc.T,Pi)

        Diff = PiNew - Pi

        for i in range(2*(N+1)):
            DiffNorm = DiffNorm + np.absolute(Diff[i])

        Pi = PiNew


    PageRank_Oracle = Pi

    PageRank_Off = np.zeros((N,1),dtype = float)

    PageRank_Def = np.zeros((N,1),dtype = float)

    for k in range(N):
        PageRank_Off[k] = PageRank_Oracle[k]

    for k in range(N+1,2*N+1):
        PageRank_Def[k-(N+1)] = PageRank_Oracle[k]                


    PageRank_Off = (1/np.sum(PageRank_Off))*PageRank_Off
    PageRank_Def = (1/np.sum(PageRank_Def))*PageRank_Def

    return PageRank_Off,PageRank_Def




def Training_Set_ncaabwalkod(Data_Full,Lines,schedule,HomeAway,day,S_OffDef_stack,feature_node2vec,height,node2vec_dim):

    N = Data_Full.shape[2]

    gamecount_train = 0
    
    y_train = np.zeros((6500,2),dtype = float)
    last_5_train = np.zeros((6500,10),dtype = float)


    x_train_home = np.zeros((6500,2*height*node2vec_dim),dtype = float)
    x_train_away = np.zeros((6500,2*height*node2vec_dim),dtype = float)

    one_hot = np.zeros((6500,2*N),dtype = float)


    for k in range(N):
        for j in range(day):
            if HomeAway[k,j] == 2:
                gamecount_train = gamecount_train + 1
                opponent = schedule[k,j]
                game = Data_Full[j,0,k]

                #Convolution Operation as described in #Atwood, Towsley, Diffusion Convolutional Neural Network, November 15, 2015
                #arXiv:1511.02136v6 [cs.LG]

                Q_h_off = S_OffDef_stack[:,k,:].T
                Q_a_def = S_OffDef_stack[:,opponent+N+1,:].T


                Q_a_off = S_OffDef_stack[:,opponent,:].T
                Q_h_def = S_OffDef_stack[:,k+N+1,:].T


                h_sheet_off = np.matmul(Q_h_off,feature_node2vec)
                a_sheet_def = np.matmul(Q_a_def,feature_node2vec)


                a_sheet_off = np.matmul(Q_a_off,feature_node2vec)
                h_sheet_def = np.matmul(Q_h_def,feature_node2vec)



                h_vec_off = h_sheet_off.flatten()
                a_vec_def = a_sheet_def.flatten()

                a_vec_off = a_sheet_off.flatten()
                h_vec_def = h_sheet_def.flatten() 

                     

                h_vec_off = h_vec_off.T
                a_vec_def = a_vec_def.T

                a_vec_off = a_vec_off.T
                h_vec_def = h_vec_def.T



                h_vec = np.concatenate((h_vec_off,a_vec_def),axis = -1)
                a_vec = np.concatenate((a_vec_off,h_vec_def),axis = -1)


                        
                x_train_home[gamecount_train-1,:] = h_vec
                x_train_away[gamecount_train-1,:] = a_vec

                y_train[gamecount_train-1] = [Data_Full[j,8,k],Data_Full[j,9,k]]

                one_hot[gamecount_train-1,k] = 1
                one_hot[gamecount_train-1,opponent+N]


                if game > 5:
                    last_5 = Data_Full[(j-6):(j-1),0,k]
                    last_5_opp = Data_Full[(j-6):(j-1),0,opponent]

                    for q in range(5):
                        if last_5[q] != 0:
                            last_5[q] = 1
                        if last_5_opp[q] != 0:
                            last_5_opp[q] = 1


                    last_5_train[gamecount_train-1] = np.concatenate((last_5,last_5_opp),axis = -1)




    x_train_home = x_train_home[0:gamecount_train,:]
    x_train_away = x_train_away[0:gamecount_train,:]



    y_train = y_train[0:gamecount_train,:]
    last_5_train = last_5_train[0:gamecount_train,:]
    one_hot = one_hot[0:gamecount_train]




    x_train = np.zeros((gamecount_train,2,2*height*node2vec_dim),dtype = float)

    for c in range(gamecount_train):
        x_train[c,0,:] = x_train_home[c,:]
        x_train[c,1,:] = x_train_away[c,:]



    return x_train, y_train,last_5_train,one_hot




def Test_Set_ncaabwalkod(Data_Full,games,testgamecount,S_OffDef_stack,feature_node2vec,height,node2vec_dim,day,year):

    N = Data_Full.shape[2]

 
    x_test_home = np.zeros((testgamecount,2*height*node2vec_dim),dtype = float)
    x_test_away = np.zeros((testgamecount,2*height*node2vec_dim),dtype = float)


    last_5_test = np.zeros((testgamecount,10),dtype = float)

    test_y = np.zeros((testgamecount,2),dtype = float)

    one_hot = np.zeros((testgamecount,2*N),dtype = float)


    for i in range(testgamecount):

        Q_h_off_t = S_OffDef_stack[:,games[i,0],:].T
        Q_a_def_t = S_OffDef_stack[:,games[i,1]+N+1,:].T


        Q_a_off_t = S_OffDef_stack[:,games[i,1],:].T
        Q_h_def_t = S_OffDef_stack[:,games[i,0]+N+1,:].T


        h_sheet_off_t = np.matmul(Q_h_off_t,feature_node2vec)
        a_sheet_def_t = np.matmul(Q_a_def_t,feature_node2vec)


        a_sheet_off_t = np.matmul(Q_a_off_t,feature_node2vec)
        h_sheet_def_t = np.matmul(Q_h_def_t,feature_node2vec)



        h_vec_off_t = h_sheet_off_t.flatten()
        a_vec_def_t = a_sheet_def_t.flatten()

        a_vec_off_t = a_sheet_off_t.flatten()
        h_vec_def_t = h_sheet_def_t.flatten() 


                      
        h_vec_off_t = h_vec_off_t.T
        a_vec_def_t = a_vec_def_t.T

        a_vec_off_t = a_vec_off_t.T
        h_vec_def_t = h_vec_def_t.T




        h_vec_t = np.concatenate((h_vec_off_t,a_vec_def_t),axis = -1)
        a_vec_t = np.concatenate((a_vec_off_t,h_vec_def_t),axis = -1)


        x_test_home[i,:] = h_vec_t
        x_test_away[i,:] = a_vec_t

        one_hot[i,games[i,0]] = 1
        one_hot[i,games[i,1]+N] = 1


        if year < 2021:
            test_y[i] = [games[i,4],games[i,5]]



        if games[i,6] > 5:
            last_5 = Data_Full[(day-5):day,0,games[i,0]]
            last_5_opp = Data_Full[(day-5):day,0,games[i,1]]

            for q in range(5):
                if last_5[q] != 0:
                    last_5[q] = 1
                if last_5_opp[q] != 0:
                    last_5_opp[q] = 1


            last_5_test[i,:] = np.concatenate((last_5,last_5_opp),axis = -1)


    x_test = np.zeros((testgamecount,2,2*height*node2vec_dim),dtype = float)

    for c in range(testgamecount):
        x_test[c,0,:] = x_test_home[c,:]
        x_test[c,1,:] = x_test_away[c,:]


    return x_test, last_5_test, test_y,one_hot







def Test_Games(TeamList,Data_Full,schedule,HomeAway,Lines,day):

    N = Data_Full.shape[2]

    games = np.zeros((N,7),dtype = object)


    testgamecount = 0
            
    for j in range(N):
        if HomeAway[j,day+1] == 2:

            testgamecount = testgamecount + 1
            opp = schedule[j,day+1]
            home_score_act = Data_Full[day+1,8,j]
            away_score_act = Data_Full[day+1,9,j]
            week_test = Data_Full[day+1,0,j]
            games[testgamecount-1,:] = [j,opp,0,0,home_score_act,away_score_act,week_test]
                    

            
    games = games[~np.all(games == 0, axis=1)]


    gameteams = np.zeros((testgamecount,2),dtype = object)

    for k in range(testgamecount):
        gameteams[k,0] = TeamList[games[k,0]][0]
        gameteams[k,1] = TeamList[games[k,1]][0]

    return games,gameteams, testgamecount



def GAT_training_set(Data_Full,Lines,schedule,HomeAway,day,feature_node2vec,A_OffDef):

    N = Data_Full.shape[2]

    gamecount_train = 0

    x_train = np.zeros((6500,2),dtype = int)
    x_train_AH = np.zeros((6500,2),dtype = int)
    y_train = np.zeros((6500,2),dtype = float)
    feature_train = np.zeros((6500,feature_node2vec.shape[0],feature_node2vec.shape[1]),dtype = float)
    A_Train = np.zeros((6500,A_OffDef.shape[0],A_OffDef.shape[1]),dtype = int)
    last_5_train = np.zeros((6500,10),dtype = float)
    one_hot = np.zeros((6500,2*N),dtype = float)
    one_hot_AH = np.zeros((6500,2*N),dtype = float)

    #A GAT representation will be computed for each node in both SOffDef and the Vegas graphs


    for k in range(N):
        for j in range(day):
            if HomeAway[k,j] == 2:

                gamecount_train = gamecount_train + 1
                opponent = schedule[k,j]
                game = Data_Full[j,0,k]
                x_train[gamecount_train-1,0] = k
                x_train[gamecount_train-1,1] = int(opponent)

                x_train_AH[gamecount_train-1,0] = int(opponent)
                x_train_AH[gamecount_train-1,1] = k

                y_train[gamecount_train-1,:] = [Data_Full[j,8,k],Data_Full[j,9,k]]

                one_hot[gamecount_train-1,k] = 1
                one_hot[gamecount_train-1,opponent+N]

                one_hot_AH[gamecount_train-1,opponent] = 1
                one_hot_AH[gamecount_train-1,k+N] = 1


                A_Train[gamecount_train-1,:,:] = A_OffDef
                A_Train[gamecount_train-1,k,opponent+N+1] = 0
                A_Train[gamecount_train-1,k+N+1,opponent] = 0
                A_Train[gamecount_train-1,opponent,k+N+1] = 0
                A_Train[gamecount_train-1,opponent+N+1,k] = 0
                feature_train[gamecount_train-1,:,:] = feature_node2vec

                if game > 5:
                    last_5 = Data_Full[(j-6):(j-1),0,k]
                    last_5_opp = Data_Full[(j-6):(j-1),0,opponent]

                    for q in range(5):
                        if last_5[q] != 0:
                            last_5[q] = 1
                        if last_5_opp[q] != 0:
                            last_5_opp[q] = 1


                    last_5_train[gamecount_train-1,:] = np.concatenate((last_5,last_5_opp),axis = -1)

                

                


    y_train = y_train[0:gamecount_train,:]
    x_train = x_train[0:gamecount_train,:]
    x_train_AH =  x_train_AH[0:gamecount_train,:]
    A_Train = A_Train[0:gamecount_train,:,:]
    feature_train = feature_train[0:gamecount_train,:,:]
    last_5_train = last_5_train[0:gamecount_train,:]
    one_hot = one_hot[0:gamecount_train]
    one_hot_AH = one_hot_AH[0:gamecount_train]


    return x_train,x_train_AH,y_train,feature_train,A_Train,last_5_train,one_hot,one_hot_AH

def gin_training_set(Data_Full,schedule,HomeAway,day,feature_node2vec,A_OffDef):

    N = Data_Full.shape[2]

    gamecount_train = 0

    x_train = np.zeros((6500,2),dtype = int)
    x_train_AH = np.zeros((6500,2),dtype = int)
    y_train = np.zeros((6500,2),dtype = float)
    feature_train = np.zeros((6500,feature_node2vec.shape[0],feature_node2vec.shape[1]),dtype = float)
    A_Train = np.zeros((6500,A_OffDef.shape[0],A_OffDef.shape[1]),dtype = int)
    last_5_train = np.zeros((6500,10),dtype = float)
    one_hot = np.zeros((6500,2*N),dtype = float)
    one_hot_AH = np.zeros((6500,2*N),dtype = float)


    #A GAT representation will be computed for each node in both SOffDef and the Vegas graphs


    for k in range(N):
        for j in range(day):
            if HomeAway[k,j] == 2:

                gamecount_train = gamecount_train + 1
                opponent = schedule[k,j]
                game = Data_Full[j,0,k]
                x_train[gamecount_train-1,0] = k
                x_train[gamecount_train-1,1] = int(opponent)

                x_train_AH[gamecount_train-1,0] = int(opponent)
                x_train_AH[gamecount_train-1,1] = k

                y_train[gamecount_train-1,:] = [Data_Full[j,8,k],Data_Full[j,9,k]]

                one_hot[gamecount_train-1,k] = 1
                one_hot[gamecount_train-1,opponent+N]

                one_hot_AH[gamecount_train-1,opponent] = 1
                one_hot_AH[gamecount_train-1,k+N] = 1


                A_Train[gamecount_train-1,:,:] = A_OffDef
                A_Train[gamecount_train-1,k,opponent+N+1] = 0
                A_Train[gamecount_train-1,k+N+1,opponent] = 0
                A_Train[gamecount_train-1,opponent,k+N+1] = 0
                A_Train[gamecount_train-1,opponent+N+1,k] = 0
                feature_train[gamecount_train-1,:,:] = feature_node2vec

                if game > 5:
                    last_5 = Data_Full[(j-6):(j-1),0,k]
                    last_5_opp = Data_Full[(j-6):(j-1),0,opponent]

                    for q in range(5):
                        if last_5[q] != 0:
                            last_5[q] = 1
                        if last_5_opp[q] != 0:
                            last_5_opp[q] = 1


                    last_5_train[gamecount_train-1,:] = np.concatenate((last_5,last_5_opp),axis = -1)

                

                


    y_train = y_train[0:gamecount_train,:]
    x_train = x_train[0:gamecount_train,:]
    x_train_AH = x_train_AH[0:gamecount_train,:]
    A_Train = A_Train[0:gamecount_train,:,:]
    feature_train = feature_train[0:gamecount_train,:,:]
    last_5_train = last_5_train[0:gamecount_train,:]
    one_hot = one_hot[0:gamecount_train]
    one_hot_AH = one_hot[0:gamecount_train]


    return x_train,y_train,feature_train,A_Train,last_5_train,one_hot

def GAT_test_set(Data_Full,games,testgamecount,feature_node2vec,A_OffDef,day,year):

    N = Data_Full.shape[2]

    feature_test = np.zeros((testgamecount,feature_node2vec.shape[0],feature_node2vec.shape[1]),dtype = float)

    A_Test = np.zeros((testgamecount,A_OffDef.shape[0],A_OffDef.shape[1]),dtype = int)

    x_test = np.zeros((testgamecount,2),dtype = float)
    x_test_AH = np.zeros((testgamecount,2),dtype = float)
    last_5_test = np.zeros((testgamecount,10),dtype = float)
    test_y = np.zeros((testgamecount,2),dtype = float)
    one_hot = np.zeros((testgamecount,2*N),dtype = float)
    one_hot_AH = np.zeros((testgamecount,2*N),dtype = float)

    for i in range(testgamecount):
        x_test[i,0] = games[i,0]
        x_test[i,1] = games[i,1]

        x_test_AH[i,0] = games[i,1]
        x_test_AH[i,1] = games[i,0]

        one_hot[i,games[i,0]] = 1
        one_hot[i,games[i,1]+N] = 1

        one_hot_AH[i,games[i,1]] = 1
        one_hot_AH[i,games[i,0]+N] = 1

        if year < 2021:
            test_y[i,:] = [games[i,4],games[i,5]]

        A_Test[i,:,:] = A_OffDef
        feature_test[i,:,:] = feature_node2vec

        if games[i,6] > 5:
            last_5 = Data_Full[(day-5):day,0,games[i,0]]
            last_5_opp = Data_Full[(day-5):day,0,games[i,1]]

            for q in range(5):
                if last_5[q] != 0:
                    last_5[q] = 1
                if last_5_opp[q] != 0:
                    last_5_opp[q] = 1


            last_5_test[i,:] = np.concatenate((last_5,last_5_opp),axis = -1)




    return x_test,x_test_AH,feature_test,A_Test,last_5_test, test_y,one_hot,one_hot_AH

def gin_test_set(Data_Full,games,testgamecount,feature_node2vec,A_OffDef,day,year):

    N = Data_Full.shape[2]

    feature_test = np.zeros((testgamecount,feature_node2vec.shape[0],feature_node2vec.shape[1]),dtype = float)

    A_Test = np.zeros((testgamecount,A_OffDef.shape[0],A_OffDef.shape[1]),dtype = int)

    x_test = np.zeros((testgamecount,2),dtype = float)
    x_test_AH = np.zeros((testgamecount,2),dtype = float)
    last_5_test = np.zeros((testgamecount,10),dtype = float)
    test_y = np.zeros((testgamecount,),dtype = float)
    one_hot = np.zeros((testgamecount,2*N),dtype = float)
    one_hot_AH = np.zeros((testgamecount,2*N),dtype = float)


    for i in range(testgamecount):
        x_test[i,0] = games[i,0]
        x_test[i,1] = games[i,1]

        x_test_AH[i,0] = games[i,1]
        x_test_AH[i,1] = games[i,0]

        one_hot[i,games[i,0]] = 1
        one_hot[i,games[i,1]+N] = 1

        one_hot_AH[i,games[i,1]] = 1
        one_hot_AH[i,games[i,0]+N] = 1

        if year < 2021:
            test_y[i] = [games[i,4],games[i,5]]

        A_Test[i,:,:] = A_OffDef
        feature_test[i,:,:] = feature_node2vec

        if games[i,6] > 5:
            last_5 = Data_Full[(day-5):day,0,games[i,0]]
            last_5_opp = Data_Full[(day-5):day,0,games[i,1]]

            for q in range(5):
                if last_5[q] != 0:
                    last_5[q] = 1
                if last_5_opp[q] != 0:
                    last_5_opp[q] = 1


            last_5_test[i,:] = np.concatenate((last_5,last_5_opp),axis = -1)

    return x_test,x_test_AH,feature_test,A_Test,last_5_test,test_y,one_hot,one_hot_AH


def Vegas_Graph(schedule,Lines,day):

    N = schedule.shape[0]

    Vegas_Graph = np.zeros((N+1,N+1),dtype = float)

    for i in range(N):
        for j in range(day):

            if schedule[i,j] != -1:
                opp = schedule[i,j]

                if Lines[i,j] < 0:
                    Vegas_Graph[i,opp] = Vegas_Graph[i,opp] + -1*Lines[i,j]

    for i in range(N):
        for j in range(N):
            gc = 0
            for k in range(day):
                if schedule[i,k] == j:
                    gc = gc + 1

            if gc != 0:
                Vegas_Graph[i,j] = Vegas_Graph[i,j]/gc


    for i in range(N):
        if np.sum(Vegas_Graph[i,:]) != 0:
            Vegas_Graph[i,:] = (1/np.sum(Vegas_Graph[i,:]))*Vegas_Graph[i,:]



    for r in range(N):
        nonzerocount = 0
        for col in range(N):
            if Vegas_Graph[r,col] != 0:
                nonzerocount = nonzerocount + 1

        if nonzerocount != 0:
            Vegas_Graph[r,N] = 1/nonzerocount
            Vegas_Graph[r,:] = 1/(np.sum(Vegas_Graph[r,:]))*Vegas_Graph[r,:]

        elif nonzerocount == 0:
            Vegas_Graph[r,N] = 1


    for c in range(N):
        Vegas_Graph[N,c] = np.sum(Vegas_Graph[:,c])


    Vegas_Graph[N,:] = 1/(np.sum(Vegas_Graph[N,:]))*Vegas_Graph[N,:]

    return Vegas_Graph


def tourney_set_ncaabwalkod(TeamList,S_OffDef_stack,feature_node2vec,height,node2vec_dim):

    N = TeamList.shape[0]

    games = pd.read_excel('data/tourney_games8.xls',sheet_name = 0,header = 0)
    games = games.to_numpy(dtype = object,copy = True)

    for i in range(games.shape[0]):

        for j in range(N):
            if games[i,0] == TeamList[j]:
                games[i,0] = j
                break

        for k in range(N):
            if games[i,1] == TeamList[k]:
                games[i,1] = k
                break




    x_test_home = np.zeros((games.shape[0],2*height*node2vec_dim),dtype = float)
    x_test_away = np.zeros((games.shape[0],2*height*node2vec_dim),dtype = float)


    last_5_test = np.zeros((games.shape[0],10),dtype = float)

    test_y = np.zeros((games.shape[0],),dtype = float)

    one_hot = np.zeros((games.shape[0],2*N),dtype = float)


    for i in range(games.shape[0]):

        Q_h_off_t = S_OffDef_stack[:,games[i,0],:].T
        Q_a_def_t = S_OffDef_stack[:,games[i,1]+N+1,:].T


        Q_a_off_t = S_OffDef_stack[:,games[i,1],:].T
        Q_h_def_t = S_OffDef_stack[:,games[i,0]+N+1,:].T


        h_sheet_off_t = np.matmul(Q_h_off_t,feature_node2vec)
        a_sheet_def_t = np.matmul(Q_a_def_t,feature_node2vec)


        a_sheet_off_t = np.matmul(Q_a_off_t,feature_node2vec)
        h_sheet_def_t = np.matmul(Q_h_def_t,feature_node2vec)



        h_vec_off_t = h_sheet_off_t.flatten()
        a_vec_def_t = a_sheet_def_t.flatten()

        a_vec_off_t = a_sheet_off_t.flatten()
        h_vec_def_t = h_sheet_def_t.flatten() 


                      
        h_vec_off_t = h_vec_off_t.T
        a_vec_def_t = a_vec_def_t.T

        a_vec_off_t = a_vec_off_t.T
        h_vec_def_t = h_vec_def_t.T




        h_vec_t = np.concatenate((h_vec_off_t,a_vec_def_t),axis = -1)
        a_vec_t = np.concatenate((a_vec_off_t,h_vec_def_t),axis = -1)


        x_test_home[i,:] = h_vec_t
        x_test_away[i,:] = a_vec_t

        one_hot[i,games[i,0]] = 1
        one_hot[i,games[i,1]+N] = 1




    x_test = np.zeros((games.shape[0],2,2*height*node2vec_dim),dtype = float)

    for c in range(games.shape[0]):
        x_test[c,0,:] = x_test_home[c,:]
        x_test[c,1,:] = x_test_away[c,:]


    return x_test,last_5_test,one_hot

def tourney_set_GAT(TeamList,feature_node2vec,A_OffDef):

    N = TeamList.shape[0]


    games = pd.read_excel('data/tourney_games8.xls',sheet_name = 0,header = 0)
    games = games.to_numpy(dtype = object,copy = True)

    for i in range(games.shape[0]):

        for j in range(N):
            if games[i,0] == TeamList[j]:
                games[i,0] = j
                break

        for k in range(N):
            if games[i,1] == TeamList[k]:
                games[i,1] = k
                break

    feature_test = np.zeros((games.shape[0],feature_node2vec.shape[0],feature_node2vec.shape[1]),dtype = float)

    A_Test = np.zeros((games.shape[0],A_OffDef.shape[0],A_OffDef.shape[1]),dtype = int)

    x_test = np.zeros((games.shape[0],2),dtype = float)
    x_test_AH = np.zeros((games.shape[0],2),dtype = float)
    last_5_test = np.zeros((games.shape[0],10),dtype = float)
    one_hot = np.zeros((games.shape[0],2*N),dtype = float)
    one_hot_AH = np.zeros((games.shape[0],2*N),dtype = float)




    for i in range(games.shape[0]):
        x_test[i,0] = games[i,0]
        x_test[i,1] = games[i,1]

        x_test_AH[i,0] = games[i,1]
        x_test_AH[i,1] = games[i,0]

        one_hot[i,games[i,0]] = 1
        one_hot[i,games[i,1]+N] = 1

        one_hot_AH[i,games[i,1]] = 1
        one_hot_AH[i,games[i,0]+N] = 1

        A_Test[i,:,:] = A_OffDef
        feature_test[i,:,:] = feature_node2vec




    return x_test,x_test_AH,feature_test,A_Test,last_5_test,one_hot,one_hot_AH


def tourney_set_gin(TeamList,feature_node2vec,A_OffDef):

    N = TeamList.shape[0]


    games = pd.read_excel('data/tourney_games8.xls',sheet_name = 0,header = 0)
    games = games.to_numpy(dtype = object,copy = True)

    for i in range(games.shape[0]):

        for j in range(N):
            if games[i,0] == TeamList[j]:
                games[i,0] = j
                break

        for k in range(N):
            if games[i,1] == TeamList[k]:
                games[i,1] = k
                break

    feature_test = np.zeros((games.shape[0],feature_node2vec.shape[0],feature_node2vec.shape[1]),dtype = float)

    A_Test = np.zeros((games.shape[0],A_OffDef.shape[0],A_OffDef.shape[1]),dtype = int)

    x_test = np.zeros((games.shape[0],2),dtype = float)
    x_test_AH = np.zeros((games.shape[0],2),dtype = float)
    last_5_test = np.zeros((games.shape[0],10),dtype = float)
    one_hot = np.zeros((games.shape[0],2*N),dtype = float)
    one_hot_AH = np.zeros((games.shape[0],2*N),dtype = float)




    for i in range(games.shape[0]):
        x_test[i,0] = games[i,0]
        x_test[i,1] = games[i,1]

        x_test_AH[i,0] = games[i,1]
        x_test_AH[i,1] = games[i,0]

        one_hot[i,games[i,0]] = 1
        one_hot[i,games[i,1]+N] = 1

        one_hot_AH[i,games[i,1]] = 1
        one_hot_AH[i,games[i,0]+N] = 1


        A_Test[i,:,:] = A_OffDef
        feature_test[i,:,:] = feature_node2vec




    return x_test,x_test_AH,feature_test,A_Test,last_5_test,one_hot,one_hot_AH




















                
    