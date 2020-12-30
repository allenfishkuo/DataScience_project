# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 14:03:59 2020

@author: allen
"""

import torch
import torch.nn as nn
import numpy as np
import new_dataloader
import backtest_new

import trading_period_by_gate_mean_new
#import matrix_trading
import os 
import pandas as pd
import torch
import torch.utils.data as Data

import matplotlib.pyplot as plt
import time
path_to_image = "C:/Users/Allen/pair_trading DL/negative profit of 2018/"


path_to_average = "./2018/averageprice/"
ext_of_average = "_averagePrice_min.csv"
path_to_minprice = "./2018/minprice/"
ext_of_minprice = "_min_stock.csv"

path_to_2015compare = "./newstdcompare2015/" 
path_to_2016compare = "./newstdcompare2016/" 
path_to_2017compare = "./newstdcompare2017/" 
path_to_2018compare = "./newstdcompare2018/" 
ext_of_compare = "_table.csv"


path_to_half = "C:/Users/Allen/pair_trading DL2/2016/2016_half/"
path_to_2017half = "./2017_halfmin/"
path_to_2018half = "./2018_halfmin/"
ext_of_half = "_half_min.csv"

path_to_profit = "./profit/wavenet2018/"

max_posion = 5

def test_reward():
    total_reward = 0
    total_num = 0
    total_trade =[0,0,0]
    action_list=[]
    check = 0
    actions = [[0.5,2.5],[1.0,3.0],[1.5,3.5],[2.0,4.0],[2.5,4.5],[3.0,5.0]]    
    Net = torch.load('2015-2016_6action.pkl')
    Net.eval()
    #print(Net)
    whole_year = new_dataloader.test_data()
    whole_year = torch.FloatTensor(whole_year).cuda()
    #print(whole_year)
    torch_dataset_train = Data.TensorDataset(whole_year)
    whole_test = Data.DataLoader(
            dataset=torch_dataset_train,      # torch TensorDataset format
            batch_size = 128,      # mini batch size
            shuffle = False,               
            )
    for step, (batch_x,) in enumerate(whole_test):
        #print(batch_x)
        output = Net(batch_x)               # cnn output
        _, predicted = torch.max(output, 1)
        action_choose = predicted.cpu().numpy()
        action_choose = action_choose.tolist()
        action_list.append(action_choose)
   # action_choose = predicted.cpu().numpy()
    action_list =sum(action_list, [])
    print(len(action_list))

    
    count_test = 0
    datelist = [f.split('_')[0] for f in os.listdir(path_to_2018compare)]
    #print(datelist[167:])
    profit_count = 0
    for date in sorted(datelist[:]): #決定交易要從何時開始
        table = pd.read_csv(path_to_2018compare+date+ext_of_compare)
        mindata = pd.read_csv(path_to_average+date+ext_of_average)
        try:
            tickdata = pd.read_csv(path_to_minprice+date+ext_of_minprice).drop([266, 267, 268, 269, 270])
        except:
            continue
        #halfmin = pd.read_csv(path_to_half+date+ext_of_half)
        #print(tickdata.shape)
        #tickdata = tickdata.iloc[166:]
        #tickdata.index = np.arange(0,len(tickdata),1)  
        num = np.arange(0,len(table),1)
        #print(date)
        for pair in num: #看table有幾列配對 依序讀入
            #action_choose = 0
            #try :
           #     spread =  table.w1[pair] * np.log(mindata[ str(table.stock1[pair]) ]) + table.w2[pair] * np.log(mindata[ str(table.stock2[pair]) ])
            ##    continue

            for i in range(6):
                #print(action_list[count_test])
                if action_list[count_test] == i :
                    open, loss = actions[i][0], actions[i][1] 
                    #open, loss = 1.5, 20000#
            choose_date = date[0:4]+'-'+date[4:6]+'-'+date[6:8]
            #print(choose_date)
            profit,opennum,trade_capital,trading  = backtest_new.pairs( choose_date,pair ,166,table,  open , loss ,tickdata, max_posion , 0.0015, 0.0015 , 300000000 )
            #print(trading)
            if profit > 0 and opennum == 1 :
                profit_count +=1
                print("有賺錢的pair",profit)
                """
                flag = os.path.isfile(path_to_profit+str(date)+'_profit.csv')
        
                if not flag :
                    df = pd.DataFrame({"stock1":[table.stock1[pair]],"stock2":[table.stock2[pair]],"trade_capital":[trade_capital],"open":[open],"loss":[loss],"reward":[profit],"open_num":[opennum]})
                    df.to_csv(path_to_profit+str(date)+'_profit.csv', mode='w',index=False)
                else :
                    df = pd.DataFrame({"stock1":[table.stock1[pair]],"stock2":[table.stock2[pair]],"trade_capital":[trade_capital],"open":[open],"loss":[loss],"reward":[profit],"open_num":[opennum]})
                    df.to_csv(path_to_profit+str(date)+'_profit.csv', mode='a', header=False,index=False)  
                """
                
                
            elif opennum ==1 and profit < 0 :
                
                print("賠錢的pair :", profit)
                """
                flag = os.path.isfile(path_to_profit+str(date)+'_profit.csv')
        
                if not flag :
                    df = pd.DataFrame({"stock1":[table.stock1[pair]],"stock2":[table.stock2[pair]],"trade_capital":[trade_capital],"open":[open],"loss":[loss],"reward":[profit],"open_num":[opennum]})
                    df.to_csv(path_to_profit+str(date)+'_profit.csv', mode='w',index=False)
                else :
                    df = pd.DataFrame({"stock1":[table.stock1[pair]],"stock2":[table.stock2[pair]],"trade_capital":[trade_capital],"open":[open],"loss":[loss],"reward":[profit],"open_num":[opennum]})
                    df.to_csv(path_to_profit+str(date)+'_profit.csv', mode='a', header=False,index=False) 
                """
                
            #print("開倉次數 :",opennum)
 
            if opennum == 1 or opennum == 0:
                check += 1
                
            total_reward += profit            
            total_num += opennum
            count_test +=1
            total_trade[0] += trading[0]
            total_trade[1] += trading[1]
            total_trade[2] += trading[2]
            
            
    print("total :",check)        
            #print(count_test)
    print("利潤  and 開倉次數 and 開倉有賺錢的次數/開倉次數:",total_reward ,total_num, profit_count/total_num)
    print("開倉有賺錢次數 :",profit_count)
    print("正常平倉 停損平倉 強迫平倉 :",total_trade[0],total_trade[1],total_trade[2])
    