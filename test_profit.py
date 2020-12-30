# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:23:13 2020

@author: allen
"""
import torch
import torch.nn as nn
import numpy as np


#import trading_period_by_test
import trading_period_by_gate_mean_new

import os 
import pandas as pd
import torch
import torch.utils.data as Data
#from DL_trading import CNN_classsification1
import matplotlib.pyplot as plt
path_to_image = "C:/Users/Allen/pair_trading DL2/trading_image_2016/"
path_to_average = "./2016/averageprice/"
ext_of_average = "_averagePrice_min.csv"

path_to_minprice = "./2016/minprice/"
ext_of_minprice = "_min_stock.csv"

path_to_2015compare = "./newstdcompare2015/" 
path_to_2016compare = "./newstdcompare2016/" 
path_to_2017compare = "./newstdcompare2017/" 
path_to_2018compare = "./newstdcompare2018/"
ext_of_compare = "_table.csv"

path_to_python ="C:/Users/Allen/pair_trading DL2"

path_to_2016half = "./2016_halfmin/"
path_to_2017half = "./2017_halfmin/"
path_to_2018half = "./2018_halfmin/"
ext_of_half = "_half_min.csv"

max_posion = 5

path_to_profit = "C:/Users/Allen/pair_trading DL2/single state/"
def reward(action_list):
    total_reward = 0
    
    profit_count = 0
    all_profit = 0
    number_of_kmeans = 6
    best_trading_cost = 0
    actions = [[0.5,2.5],[1.0,3.0],[1.5,3.5],[2.0,4.0],[2.5,4.5],[3.0,5.0]]

    
    datelist = [f.split('_')[0] for f in os.listdir(path_to_2016compare)]
    profit_num = 0
    profit_count = 0
    for trading_cost in np.arange(0,0.001,0.001):
        #print("trading_cost :",trading_cost)
        cur_profit = 0
        cur_profit_count = 0
        count_test = 0
        total_num = 0
        for date in sorted(datelist): #決定交易要從何時開始
            if date[0:6] == "201611" or date[0:6] =="201612":
                table = pd.read_csv(path_to_2016compare+date+ext_of_compare)
                mindata = pd.read_csv(path_to_average+date+ext_of_average)
                tickdata = pd.read_csv(path_to_minprice+date+ext_of_minprice)
                #halfmin = pd.read_csv(path_to_half+date+ext_of_half)
                #print(tickdata.shape)
                tickdata = tickdata.iloc[166:]
                tickdata.index = np.arange(0,len(tickdata),1)    
                num = np.arange(0,len(table),1)
                #total_num += len(table)
                for pair in num: #看table有幾列配對 依序讀入
        
                    #action_choose = 0
                    #spread =  table.w1[pair] * np.log(mindata[ str(table.stock1[pair]) ]) + table.w2[pair] * np.log(mindata[ str(table.stock2[pair]) ])
                    #spread = spread.T.to_numpy()
                    #print(spread)
                    Bth1 = np.ones((5,1))
                    for i in range(number_of_kmeans):
                        if action_list[count_test] == i :
                            open, loss = actions[i][0], actions[i][1] 
        
                    
                    Bth1[2][0] = table.mu[pair]
                    Bth1[0][0] = table.mu[pair] +table.stdev[pair]*loss
                    Bth1[1][0] = table.mu[pair] +table.stdev[pair]*open
                    Bth1[3][0] = table.mu[pair] -table.stdev[pair]*open
                    Bth1[4][0] = table.mu[pair] -table.stdev[pair]*loss
                    #spread = table.w1[pair] * np.log(tickdata[ str(table.stock1[pair]) ]) + table.w2[pair] * np.log(tickdata[ str(table.stock2[pair]) ])
                    #print(Bth1)
                    
                    profit, opennum,trade_capital,_  = trading_period_by_gate_mean_new.pairs( pair ,166,  table , mindata , tickdata , open ,open, loss ,mindata, max_posion , 0.0015, trading_cost , 300000000 )
    
                    cur_profit += profit
                    if profit > 0 :
                        cur_profit_count +=1
                    count_test +=1
                    total_num += 1
                    
                
        if cur_profit > all_profit :
            all_profit = cur_profit
            profit_count = cur_profit_count
            best_trading_cost = trading_cost
            profit_num = total_num
        
                
    print("有賺錢的",profit_count)
    print("總共幾個:",total_num)
    print("action list",len(action_list))
    print("利潤 :",all_profit)
    print("best_trading_cost :",best_trading_cost)
    
    #print("利潤  and 開倉次數 and 開倉有賺錢的次數/開倉次數:",total_reward ,total_num, profit_count/total_num)
    #print("開倉有賺錢次數 :",profit_count)
    return (profit_count/total_num)*100
    

