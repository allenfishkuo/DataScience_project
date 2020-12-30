# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 19:54:16 2020

@author: Allen
"""
import numpy as np
import time
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import trading_period_by_gate_mean_new
import csv
import random
path_to_average = "./2015/averageprice/"
ext_of_average = "_averagePrice_min.csv"
path_to_minprice = "./2015/minprice/"
ext_of_minprice = "_min_stock.csv"
path_to_compare = "./newstdcompare2015/"
ext_of_compare = "_table.csv"


path_to_python ="C:/Users/Allen/pair_trading_DL4"
path=os.path.dirname(os.path.abspath(__file__))+'/results/'
path_to_check ='./check_kmean/'
path_to_20action ='C:/Users/Allen/pair_trading DL/20action kmean2018/'
path_to_20actionwithtax = 'C:/Users/Allen/pair_trading DL2/20action kmean2018 with tax/'
path_to_choose2015 = "./35newgt2018/"
path_to_all_gt = "./gt_21action_1211/"
ext_of_groundtruth = "_ground truth.csv"
max_posion = 5


reward_list=[]
datelist = [f.split('_')[0] for f in os.listdir(path_to_compare)]
print(datelist)
lower_bound = np.arange(0.5,8,0.05)
upper_bound = np.arange(5,25,1)
"""
def choose_action(lower_bound,upper_bound) :
    action_list=[]
    count = 0
    l , u = 1,0
    while count < 300 :
        l = np.random.choice(lower_bound,1)
        u = np.random.choice(upper_bound,1)        
        if 1.5*l < u :
            w = np.concatenate((l,u),axis = None)
            w = list(w)
            #print(w)
            action_list.append(w)
            count +=1
    return action_list
    
action_list = choose_action(lower_bound, upper_bound)
actions = sorted(action_list, key = lambda s: s[0])
print(actions)
"""
#actions = [[0.5002033307514231, 9.999612703330783], [0.6802278275020894, 6.970301057770593], [1.1979313380281675, 14.240316901408438], [1.238262527233124, 11.421023965141616], [1.3402498377676804, 16.44159636599612], [1.4550034387895419, 22.414030261348046], [1.7906383921974274, 19.403576178513426], [1.8334690074539046, 6.07659866614364], [1.9509871600165671, 9.182797183487537], [1.9844808743169393, 5.00000000000005], [2.263022959183676, 7.842091836734736], [2.7651543942992896, 14.789548693586722], [3.1221539283805497, 16.213254943880298], [3.350909570261011, 12.044819404165569], [3.3748820754717013, 22.837853773584932], [3.5011228070175466, 6.6417543859649335], [3.6605886116442767, 9.704414587332057], [4.388342585249806, 8.21570182394924], [4.752295918367349, 14.082417582417579], [5.4250279329609, 17.403351955307247], [6.023128205128209, 23.252307692307696], [6.186773199845984, 11.237581825182907], [7.178633217993089, 15.37456747404844], [7.83320148331276, 20.010383189122425], [7.884216491924007, 18.977897421365874]]
#actions =[[0.5, 10.0], [1.3000000000000007, 23.0], [1.3500000000000008, 6.0], [1.4000000000000008, 20.0], [1.600000000000001, 12.0], [1.6500000000000008, 16.0], [1.7500000000000009, 9.0], [1.8000000000000012, 9.0], [1.9000000000000008, 19.0], [1.9500000000000013, 5.0], [2.0000000000000013, 6.0], [2.1000000000000014, 9.0], [2.200000000000001, 6.0], [2.4000000000000017, 10.0], [2.650000000000002, 12.0], [2.7500000000000018, 15.0], [2.9000000000000017, 20.0], [3.3500000000000023, 16.0], [7.900000000000008, 20.0], [100, 2000]] # number of >800 label
actions = [[0.5,2.5],[1.0,3.0],[1.5,3.5],[2.0,4.0],[2.5,4.5],[3.0,5.0]]
print(len(actions))
for date in sorted(datelist):
    table = pd.read_csv(path_to_compare+date+ext_of_compare)
    mindata = pd.read_csv(path_to_average+date+ext_of_average)
    tickdata = pd.read_csv(path_to_minprice+date+ext_of_minprice)
    tickdata = tickdata.iloc[166:]
    tickdata.index = np.arange(0,len(tickdata),1)
    #gt = pd.read_csv(path_to_choose2018+date+ext_of_groundtruth,usecols=["action choose"])
    #gt = gt.values
    #gt= np.array(gt)
    #gt= gt.ravel()
    
    
    os.chdir(path_to_python)    
    num = np.arange(0,len(table),1)
    #gt = gt.ravel()
   # print(gt[0][0])
    for pair in num: #看table有幾列配對 依序讀入
       # print(pair)
        reward = -0.000001
        open_time = 0 
        loss_time = 0
        open_nn =0
        action_choose = 0
        #spread =  table.w1[pair] * np.log(mindata[ str(table.stock1[pair]) ]) + table.w2[pair] * np.log(mindata[ str(table.stock2[pair]) ])
        #spread = spread.T.to_numpy()
        #print(spread)
        Bth1 = np.ones((5,1))
        #print(tickdata[str(table.stock1[pair])])
        #TickTP1 = tickdata[[str(table.stock1[pair]),str(table.stock2[pair])]]
        #TickTP1 = TickTP1.T.to_numpy()
        #print(TickTP1)
        #choose = int(gt[pair])
        #open,loss = actions[gt[pair][0]]
        for open,loss in sorted(actions): #對不同開倉門檻做測試
            #print(open,loss)
            
            #Bth1[2][0] = table.mu[pair]
            #Bth1[0][0] = table.mu[pair] +table.stdev[pair]*loss
            #Bth1[1][0] = table.mu[pair] +table.stdev[pair]*open
            #Bth1[3][0] = table.mu[pair] -table.stdev[pair]*open
            #Bth1[4][0] = table.mu[pair] -table.stdev[pair]*loss
            #print(Bth1)
            profit,opennum,trade_capital,trading =  trading_period_by_gate_mean_new.pairs( pair ,166,  table , mindata , tickdata , open ,open, loss ,mindata, max_posion , 0.0015, 0 , 300000000 )
            #spread ,profit ,opennum, rt = trading_period.pairs( pair ,  table , mindata , tickdata , open , loss , max_posion , 0 , 30000000 )
            #print("利潤 :",profit)
            #print("開倉次數 :",open_num)

        
            
            if profit > reward :
                reward = profit
                open_time = open
                loss_time = loss
                open_nn = opennum
                action_ = action_choose
            if reward <= 0 :
                open_time = open
                loss_time = loss
                action_ = action_choose
                
            
            #plotB1 = np.ones((5,len(spread)))*Bth1
            action_choose +=1
            """
            plt.figure()
            plt.plot(spread)
            plt.plot(range(len(spread)),plotB1.T,"--")
            plt.title("Trade open :"+str(open)+"and loss :"+str(loss)+" stock :"+str(table.stock1[pair]) +" with "+str(table.stock2[pair]))
            #TC1fn = "Number"+str(self.count_pic)+"Pair Trading RL.png"
            #plt.savefig(path+TC1fn)
            plt.show()
            plt.close()
            """
       # print(date[0:4]) 
    #print("====================================================================================")
        flag = os.path.isfile(path_to_all_gt+str(date[0:4])+'/'+str(date)+'_ground truth.csv')
        
        if not flag :
            df = pd.DataFrame({"stock1":[table.stock1[pair]],"stock2":[table.stock2[pair]],"open":[open_time],"loss":[loss_time],"reward":[reward],"open_num":[open_nn],"action choose":[action_]})
            df.to_csv(path_to_all_gt+str(date[0:4])+'/'+str(date)+'_ground truth.csv', mode='w',index=False)
        else :
            df = pd.DataFrame({"stock1":[table.stock1[pair]],"stock2":[table.stock2[pair]],"open":[open_time],"loss":[loss_time],"reward":[reward],"open_num":[open_nn],"action choose":[action_]})
            df.to_csv(path_to_all_gt+str(date[0:4])+'/'+str(date)+'_ground truth.csv', mode='a', header=False,index=False)
            #print(P1)
            #print(C1)
            

