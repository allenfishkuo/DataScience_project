# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:23:13 2020

@author: allen
"""
import torch
import torch.nn as nn
import numpy as np
import new_dataloader


import trading_period_by_gate_mean_new
#import matrix_trading
import os 
import pandas as pd
import torch
import torch.utils.data as Data

import matplotlib.pyplot as plt
import time
path_to_image = "C:/Users/Allen/pair_trading DL/negative profit of 2018/"


path_to_average = "./2017/averageprice/"
ext_of_average = "_averagePrice_min.csv"
path_to_minprice = "./2017/minprice/"
ext_of_minprice = "_min_stock.csv"

path_to_2015compare = "./newstdcompare2015/" 
path_to_2016compare = "./newstdcompare2016/" 
path_to_2017compare = "./newstdcompare2017/" 
path_to_2018compare = "./newstdcompare2018/" 
ext_of_compare = "_table.csv"

path_to_python ="C:/Users/Allen/pair_trading DL4"

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
    #actions =[[0.5000000000001013, 1.6058835588357105], [1.1231567674441643, 3.009226460170205], [1.6774656461992412, 8.482170812315225], [2.434225143491954, 5.0301795963708305], [3.838786213786223, 7.405844155844149], [50, 100]]
    #actions =[[0.5000000000001013, 1.6058835588357105], [0.8422529644270367, 2.7302766798420457], [1.42874957000333, 3.312693498451958], [1.681668686169194, 8.472736714557769], [2.054041204437417, 4.680031695721116], [3.1352641629535314, 5.810311903246376], [4.378200155159055, 8.429014740108636], [5.632843137254913, 16.43431372549023], [6.8013888888889005, 13.081481481481516], [50,100]]
    #actions = [[0.5000000000002669, 2.500000000000112], [0.7288428324698772, 4.0090056748083995], [1.1218344155846804, 3.0000000000002496], [1.2162849872773496, 7.4631043256997405], [1.4751902346226717, 3.9999999999997113], [1.749999999999973, 3.4999999999998117], [2.086678832116794, 6.2883211678832325], [2.193017888055368, 4.018753606462444], [2.2499999999999822, 7.500000000000021], [2.6328389830508536, 8.9762711864407], [2.980046948356806, 13.515845070422579], [3.2499999999999982, 5.500000000000034], [3.453852327447829, 11.505617977528125], [3.693027210884357, 6.0739795918367605], [4.000000000000004, 12.500000000000034], [4.151949541284411, 10.021788990825703], [4.752819548872187, 15.016917293233117], [4.8633603238866225, 7.977058029689605], [5.7367647058823605, 13.470588235294136], [6.071428571428564, 16.47435897435901], [6.408839779005503, 10.95488029465933], [7.837962962962951, 12.745370370370392], [8.772727272727282, 18.23295454545456], [9.242088607594926, 14.901898734177237], [100,200]]
   # actions = [[0.5000000000001132, 2.9999999999999174], [0.5000000000001159, 1.4999999999998659], [0.5011026964048937, 5.057090545938981], [0.5997656402578979, 1.9999999999999885], [0.7450101488498877, 2.500000000000264], [0.9957274202272546, 3.5038669551108814], [0.9986299023806298, 3.00000000000035], [0.9989438479141635, 6.000440063369089], [1.2502992817239085, 7.499301675977626], [1.3748807883861818, 5.499999999999963], [1.4714078698027613, 5.000000000000162], [1.6540920096852696, 4.3137046004844875], [1.7331017056222244, 8.988629185091602], [1.9997078301519378, 10.003506038176837], [2.131133083912334, 6.300320684126101], [2.2702554744525623, 7.199270072992672], [2.6956923890063442, 7.93093727977449], [2.9601038145823297, 8.740704973442782], [3.4156698564593317, 10.223684210526311], [4.180390032502709, 11.496208017334773], [4.475710508922673, 13.83311302048908], [5.686262376237629, 16.102103960396033], [7.189858490566042, 19.562500000000004], [9.489123012389175, 23.9905618964017], [100, 200]]
    #1-82013-2016#actions = [[1.0971083893545064, 12.780734874297824], [1.1227439921530564, 5.0000000000002025], [1.2335853575094577, 9.677591515566052], [1.2511041713139057, 15.999999999999872], [1.5038041133323725, 19.282755645045146], [1.686532170119944, 17.458312679686593], [1.9264175557936942, 7.2370125331967], [2.02304897137745, 5.627124329159388], [2.4325072358900095, 10.699589966232447], [2.5779913972888493, 13.130344108446305], [2.8119307692307682, 20.224307692307548], [2.852625298329354, 15.483160965261133], [3.2014590979045794, 9.185272877944676], [3.2373171223892623, 7.566037735848912], [3.6211730801830813, 22.306323105610964], [4.00415665989285, 11.104008867541099], [4.726152954808812, 20.078099652375336], [4.864921154418321, 12.499553704254689], [4.930584600760447, 17.123811787072206], [5.793574014481093, 14.29042638777153], [6.2467695620962, 23.51040918880109], [7.439736399326975, 19.99691531127306], [7.49569402228977, 17.16548463356975], [9.417529752331845, 23.923769700868746], [100, 2000]]
    #actions = [[0.5002033307514231, 9.999612703330783], [0.6802278275020894, 6.970301057770593], [1.1979313380281675, 14.240316901408438], [1.238262527233124, 11.421023965141616], [1.3402498377676804, 16.44159636599612], [1.4550034387895419, 22.414030261348046], [1.7906383921974274, 19.403576178513426], [1.8334690074539046, 6.07659866614364], [1.9509871600165671, 9.182797183487537], [1.9844808743169393, 5.00000000000005], [2.263022959183676, 7.842091836734736], [2.7651543942992896, 14.789548693586722], [3.1221539283805497, 16.213254943880298], [3.350909570261011, 12.044819404165569], [3.3748820754717013, 22.837853773584932], [3.5011228070175466, 6.6417543859649335], [3.6605886116442767, 9.704414587332057], [4.388342585249806, 8.21570182394924], [4.752295918367349, 14.082417582417579], [5.4250279329609, 17.403351955307247], [6.023128205128209, 23.252307692307696], [6.186773199845984, 11.237581825182907], [7.178633217993089, 15.37456747404844], [7.83320148331276, 20.010383189122425], [100, 2000]]
    #2015-2016_op 
    #actions =[[0.5, 10.0], [1.3000000000000007, 23.0], [1.3500000000000008, 6.0], [1.4000000000000008, 20.0], [1.600000000000001, 12.0], [1.6500000000000008, 16.0], [1.7500000000000009, 9.0], [1.8000000000000012, 9.0], [1.9000000000000008, 19.0], [1.9500000000000013, 5.0], [2.0000000000000013, 6.0], [2.1000000000000014, 9.0], [2.200000000000001, 6.0], [2.4000000000000017, 10.0], [2.650000000000002, 12.0], [2.7500000000000018, 15.0], [2.9000000000000017, 20.0], [3.3500000000000023, 16.0], [7.900000000000008, 20.0], [100, 2000]] # number of >800 label
    actions = [[0.5,2.5],[1.0,3.0],[1.5,3.5],[2.0,4.0],[2.5,4.5],[3.0,5.0]]
    #actions = [[0.49999999999998446, 3.5000000000000355], [0.5000000000002669, 2.500000000000112], [0.7499999999999689, 3.9999999999997025], [0.8255813953488285, 4.6886304909561005], [0.9999999999999694, 2.9999999999995994], [1.2174744897959144, 7.459183673469383], [1.24999999999997, 2.9999999999996234], [1.4751902346226717, 3.9999999999997113], [1.749999999999973, 3.4999999999998117], [1.7500000000000058, 6.499999999999994], [1.8023648648648616, 8.797297297297295], [1.9999999999999754, 4.030545112781948], [2.2499999999999822, 7.500000000000021], [2.499999999999994, 4.000000000000044], [2.4999999999999973, 6.028455284552839], [2.7500000000000036, 9.00193610842209], [2.980046948356806, 13.515845070422579], [3.2499999999999982, 5.500000000000034], [3.453852327447829, 11.505617977528125], [3.693027210884357, 6.0739795918367605], [4.000000000000004, 12.500000000000034], [4.1462703962704035, 10.038461538461554], [4.500000000000006, 7.499999999999996], [4.752819548872187, 15.016917293233117], [5.0904605263157805, 8.298245614035086], [5.526881720430115, 16.478494623655948], [5.71363636363637, 13.500000000000018], [5.964062500000008, 11.496874999999998], [6.593427835051529, 10.751288659793836], [7.252808988764048, 16.44382022471911], [7.830188679245271, 12.721698113207568], [8.8031914893617, 16.978723404255312], [8.826086956521737, 19.304347826086953], [9.22258064516128, 14.825806451612925], [100,200]]
    #print(actions[0][0])
    #Net = CNN_classsification1()
    #print(Net)
    
    Net = torch.load('2016-2016_6action.pkl')
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

    total_table = 0
    count_test = 0
    datelist = [f.split('_')[0] for f in os.listdir(path_to_2017compare)]
    #print(datelist[167:])
    profit_count = 0
    for date in sorted(datelist[:]): #決定交易要從何時開始

        table = pd.read_csv(path_to_2017compare+date+ext_of_compare)
        mindata = pd.read_csv(path_to_average+date+ext_of_average)
        total_table += len(table)
        print(total_table)
        #try:
        tickdata = pd.read_csv(path_to_minprice+date+ext_of_minprice)#.drop([266, 267, 268, 269, 270])
        #except:
            #continue
        #halfmin = pd.read_csv(path_to_half+date+ext_of_half)
        #print(tickdata.shape)
        tickdata = tickdata.iloc[166:]
        tickdata.index = np.arange(0,len(tickdata),1)  
        num = np.arange(0,len(table),1)
        print(date)
        for pair in num: #看table有幾列配對 依序讀入
            #action_choose = 0
            #try :
           #     spread =  table.w1[pair] * np.log(mindata[ str(table.stock1[pair]) ]) + table.w2[pair] * np.log(mindata[ str(table.stock2[pair]) ])
            ##    continue

            for i in range(6):
               # print(action_list[count_test])
                #print(count_test)
                if action_list[count_test] == i :
                   open, loss = actions[i][0], actions[i][1] 
                    #open, loss = 0.75, 2.5#

            profit,opennum,trade_capital,trading  = trading_period_by_gate_mean_new.pairs( pair ,166,  table , mindata , tickdata , open ,open, loss ,mindata, max_posion , 0.0015, 0.0015 , 300000000 )
            #print(trading)
            if profit > 0 and opennum == 1 :
                profit_count +=1
                #print("有賺錢的pair",profit)
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
                
                #print("賠錢的pair :", profit)
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
        print("contest",count_test)    

    print("total :",check)        
            #print(count_test)
    print("利潤  and 開倉次數 and 開倉有賺錢的次數/開倉次數:",total_reward ,total_num, profit_count/total_num)
    print("開倉有賺錢次數 :",profit_count)
    print("正常平倉 停損平倉 強迫平倉 :",total_trade[0],total_trade[1],total_trade[2])
    
          
#test()