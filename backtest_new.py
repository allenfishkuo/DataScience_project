from statsmodels.tsa.stattools import adfuller
from scipy.stats import skew
from cost import tax, slip
from integer import num_weight
from vecm import rank
from MTSA import fore_chow, spread_chow, order_select
import pandas as pd
import numpy as np
import pymysql
import os
import json
import matplotlib.pyplot as plt
db_host = '140.113.24.5'
db_name = 'Fintech'
db_user = 'fintech'
db_passwd = 'financefintech'


fin_db = pymysql.connect(
    host=db_host,
    user=db_user,
    password=db_passwd,
    db=db_name,

)
print(os.getcwd())

fin_cursor = fin_db.cursor(pymysql.cursors.DictCursor)
with open("./news_pred.json") as f:
    all_news = json.load(f)

#print(all_news)


path_to_image = './profit image/'

def pairs(choose_date, pair,formate_time,table,open,loss,tick_data, maxi,tax_cost, cost_gate,capital):
    
    s1 = str(table.stock1[pair])
    s2 = str(table.stock2[pair])

    tw1 = table.w1[pair]
    tw2 = table.w2[pair]
    Estd = table.Estd[pair] #table.Estd[pos]table.stdev[pos]
    Emu = table.Emu[pair] #table.Emu[pos]table.mu[pos]
    

    up_open_time = open
    down_open_time = up_open_time
    stop_loss_time =  loss #actions[table.action[pos]][1] #actions[0][1] 2.5

    up_open = Emu + Estd * up_open_time  # 上開倉門檻
    down_open = Emu - Estd * down_open_time  # 下開倉門檻
    stop_loss = Estd * stop_loss_time  # stop_loss_time  # 停損門檻
    close = Emu  # 平倉(均值)
    # up_open = Emu + Estd * up_open_time
    # down_open = Emu - Estd * up_open_time
     #actions[table.action[pos]][0]  1.5
    


    trade_capital = 0
    cpA, cpB = 0, 0
    trading = [0, 0, 0]

    new1_exist = 0
    new2_exist = 0
    s1_news = pd.DataFrame()
    s2_news = pd.DataFrame()
    if s1 in all_news:
        s1_news = pd.DataFrame.from_dict(all_news[s1])
        s1_news = s1_news[s1_news["time"].str.startswith(choose_date)]
    if s2 in all_news:
        s2_news = pd.DataFrame.from_dict(all_news[s2])
        s2_news = s2_news[s2_news["time"].str.startswith(choose_date)]
    

    _9am = pd.to_datetime(choose_date+" 09:00", utc=True)
    #print(s1_news.empty,s2_news.empty)
    if not s1_news.empty:
        s1_news["predict"] = s1_news["predict"].apply(lambda x: -1 if x==0 else x)
        s1_news["time"] = pd.to_datetime(s1_news["time"], utc=True)
        s1_news["9am_offset"] = s1_news["time"].apply(lambda x: (x.ceil(freq='1T') - _9am).total_seconds()//60)
        s1_news = s1_news[(s1_news["9am_offset"] <= 240) & (s1_news["9am_offset"] >= -60)]
        
        

    
    if not s2_news.empty:
        s2_news["predict"] = s2_news["predict"].apply(lambda x: -1 if x==0 else x)
        s2_news["time"] = pd.to_datetime(s2_news["time"], utc=True)
        s2_news["9am_offset"] = s2_news["time"].apply(lambda x: (x.ceil(freq='1T') - _9am).total_seconds()//60)
        s2_news = s2_news[(s2_news["9am_offset"] <= 240) & (s2_news["9am_offset"] >= -60)]


    if s1_news.empty and s2_news.empty:
        return 0, 0 ,0 ,trading

    # # 波動太小的配對不開倉
    if up_open_time * Estd < cost_gate:
        # trade_process.append([tick_data.mtimestamp[1], "配對波動太小，不開倉"])
        # print("配對波動太小，不開倉")
        trading_profit = 0
        trade = 0
        local_profit = 0
        position = 0
        return local_profit, trade_capital, trade, trading

    t = formate_time  # formate time

    local_profit = []
    
    spread_all = tw1 * np.log(tick_data[s1]) + tw2 * np.log(tick_data[s2])
    tick_data = tick_data.iloc[166:]
    tick_data.index = np.arange(0,len(tick_data),1) 
    spread = tw1 * np.log(tick_data[s1]) + tw2 * np.log(tick_data[s2])


    # M = round(1 / table.zcr[pos])  # 平均持有時間
    trade = 0  # 計算開倉次數
    break_point = 0  # 計算累積斷裂點

    position = 0  # 持倉狀態，1:多倉，0:無倉，-1:空倉，-2：強制平倉
    pos = [0]
    stock1_profit = []
    stock2_profit = []


    used_news = False

    for i in range(0, len(spread) - 2):
        if position == 0 and i != len(spread) - 3 and i < 40:  # 之前無開倉
            s1_news_trend = 0
            s2_news_trend = 0
            if not s1_news.empty:
                tmp_s1_news = s1_news[(s1_news["9am_offset"] <= (i + formate_time))].tail()
               # print(tmp_s1_news)
                if not tmp_s1_news.empty:
                    s1_news_trend = tmp_s1_news["predict"].iloc[0]
                    s1_news_title = tmp_s1_news["title"]
                    #print(s1_news_title.iloc[0])
                    s1_news_time  = tmp_s1_news["9am_offset"].iloc[0]


            if not s2_news.empty:
                tmp_s2_news = s2_news[(s2_news["9am_offset"] <= (i + formate_time))].tail()
                if not tmp_s2_news.empty:
                    s2_news_trend = tmp_s2_news["predict"].iloc[0]
            
            
            if  spread[i] < (close + stop_loss) and  spread[i] > up_open : # 碰到上開倉門檻且小於上停損門檻
                # 資金權重轉股票張數，並整數化
                if (s1_news_trend*(-tw1))<0 or (s2_news_trend*(-tw2))<0:
                    used_news = True
                    continue
                else:
                    if (s1_news_trend != 0 or s2_news_trend != 0):
                        used_news = True
                # print(tick_data.mtimestamp[i],"碰到上開倉門檻 ,上開倉")
                w1, w2 = num_weight(
                    tw1, tw2, tick_data[s1][i], tick_data[s2][i], maxi, capital)
                position = -1
                stock1_payoff = w1 * slip(tick_data[s1][i], tw1)
                stock2_payoff = w2 * slip(tick_data[s2][i], tw2)
                stock1_payoff, stock2_payoff = tax(
                    stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                cpA, cpB = stock1_payoff, stock2_payoff
                if cpA > 0 and cpB > 0:
                    trade_capital += abs(cpA) + abs(cpB)
                elif cpA > 0 and cpB < 0:
                    trade_capital += abs(cpA) + 0.9 * abs(cpB)
                elif cpA < 0 and cpB > 0:
                    trade_capital += 0.9 * abs(cpA) + abs(cpB)
                elif cpA < 0 and cpB < 0:
                    trade_capital += 0.9 * abs(cpA) + 0.9 * abs(cpB)
                    # down_open = table.mu[pos] - table.stdev[pos] * close_time
                trade += 1
                # trade_process.append([tick_data.mtimestamp[i],"碰到上開倉門檻 ,上開倉<br>",w1, w2, stock1_payoff+stock2_payoff])

            elif spread[i] > (close - stop_loss) and spread[i] < down_open:  # 碰到下開倉門檻且大於下停損門檻
                # 資金權重轉股票張數，並整數化
                if (s1_news_trend*tw1)<0 or (s2_news_trend*tw2)<0:
                    used_news = True
                    continue
                else:
                    if (s1_news_trend != 0 or s2_news_trend != 0):
                        used_news = True
                # print(tick_data.mtimestamp[i],"碰到下開倉門檻 ,下開倉")
                w1, w2 = num_weight(
                    tw1, tw2, tick_data[s1][i], tick_data[s2][i], maxi, capital)
                position = 1
                stock1_payoff = -w1 * slip(tick_data[s1][i], -tw1)
                stock2_payoff = -w2 * slip(tick_data[s2][i], -tw2)
                stock1_payoff, stock2_payoff = tax(
                    stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                cpA, cpB = stock1_payoff, stock2_payoff
                if cpA > 0 and cpB > 0:
                    trade_capital += abs(cpA) + abs(cpB)
                elif cpA > 0 and cpB < 0:
                    trade_capital += abs(cpA) + 0.9 * abs(cpB)
                elif cpA < 0 and cpB > 0:
                    trade_capital += 0.9 * abs(cpA) + abs(cpB)
                elif cpA < 0 and cpB < 0:
                    trade_capital += 0.9 * abs(cpA) + 0.9 * abs(cpB)
                # up_open = table.mu[pos] + table.stdev[pos] * close_time
                trade += 1
                # trade_process.append([tick_data.mtimestamp[i],"碰到下開倉門檻 ,下開倉<br>", -w1, -w2, stock1_payoff+stock2_payoff])
            else:
                position = 0
                stock1_payoff = 0
                stock2_payoff = 0
        elif position == -1:  # 之前有開空倉，平空倉      
            if (spread[i] - close) < 0:  # 空倉碰到下開倉門檻即平倉

                # print(tick_data.mtimestamp[i],"之前有開空倉，碰到均值，平倉")
                position = 666  # 平倉
                stock1_payoff = -w1 * slip(tick_data[s1][i], -tw1)
                stock2_payoff = -w2 * slip(tick_data[s2][i], -tw2)
                stock1_payoff, stock2_payoff = tax(
                    stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[0] += 1
                # trade_process.append([tick_data.mtimestamp[i],"碰到均值，平倉<br>",-w1, -w2, stock1_payoff+stock2_payoff])
                # down_open = table.mu[pos] - table.stdev[pos] * open_time
                # 每次交易報酬做累加(最後除以交易次數做平均)
            elif spread[i] > (close + stop_loss):  # 空倉碰到上停損門檻即平倉停損

                # print(tick_data.mtimestamp[i],"之前有開空倉，碰到上停損門檻，強制平倉")
                position = -2  # 碰到停損門檻，強制平倉
                stock1_payoff = -w1 * slip(tick_data[s1][i], -tw1)
                stock2_payoff = -w2 * slip(tick_data[s2][i], -tw2)
                stock1_payoff, stock2_payoff = tax(
                    stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[1] += 1
                # trade_process.append([tick_data.mtimestamp[i],"碰到上停損門檻，強制平倉<br>",-w1, -w2, stock1_payoff+stock2_payoff])
                # 每次交易報酬做累加(最後除以交易次數做平均)

            elif i == (len(spread) - 3):  # 回測結束，強制平倉
                # trade_process.append([tick_data.mtimestamp[i],"回測結束，強制平倉<br>"])
                # print(tick_data.mtimestamp[i],"回測結束，強制平倉")
                position = -4
                stock1_payoff = -w1 * \
                    slip(tick_data[s1][i], -tw1)
                stock2_payoff = -w2 * \
                    slip(tick_data[s2][i], -tw2)
                stock1_payoff, stock2_payoff = tax(
                    stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[2] += 1
                # trade_process.append([tick_data.mtimestamp[i],"回測結束，強制平倉<br>",-w1, -w2, stock1_payoff+stock2_payoff])
                # 每次交易報酬做累加(最後除以交易次數做平均)
            else:
                position = -1
                stock1_payoff = 0
                stock2_payoff = 0
        elif position == 1:  # 之前有開多倉，平多倉
            if (spread[i] - close) > 0:

                # print(tick_data.mtimestamp[i],"之前有開多倉，碰到均值，平倉")
                position = 666  # 平倉
                stock1_payoff = w1 * slip(tick_data[s1][i], tw1)
                stock2_payoff = w2 * slip(tick_data[s2][i], tw2)
                stock1_payoff, stock2_payoff = tax(
                    stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[0] += 1
                # trade_process.append([tick_data.mtimestamp[i],"碰到均值，平倉<br>",w1, w2, stock1_payoff+stock2_payoff])
                # up_open = table.mu[pos] + table.stdev[pos] * open_time
                # 每次交易報酬做累加(最後除以交易次數做平均)
            elif spread[i] < (close - stop_loss):

                # print(tick_data.mtimestamp[i],"之前有開多倉，碰到下停損門檻，強制平倉")
                position = -2  # 碰到停損門檻，強制平倉
                stock1_payoff = w1 * slip(tick_data[s1][i], tw1)
                stock2_payoff = w2 * slip(tick_data[s2][i], tw2)
                stock1_payoff, stock2_payoff = tax(
                    stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[1] += 1

                # trade_process.append([tick_data.mtimestamp[i],"碰到下停損門檻，強制平倉<br>", w1, w2, stock1_payoff+stock2_payoff])
                # 每次交易報酬做累加(最後除以交易次數做平均)

            elif i == (len(spread) - 3):  # 回測結束，強制平倉

                # print(tick_data.mtimestamp[i],"回測結束，強制平倉")
                position = -4
                stock1_payoff = w1 * \
                    slip(tick_data[s1][len(tick_data) - 1], tw1)
                stock2_payoff = w2 * \
                    slip(tick_data[s2][len(tick_data) - 1], tw2)
                stock1_payoff, stock2_payoff = tax(
                    stock1_payoff, stock2_payoff, position, tax_cost)  # 計算交易成本
                trading[2] += 1

                # trade_process.append([tick_data.mtimestamp[i],"回測結束，強制平倉<br>", w1, w2, stock1_payoff+stock2_payoff])
                # 每次交易報酬做累加(最後除以交易次數做平均)
            else:
                position = 1
                stock1_payoff = 0
                stock2_payoff = 0
        else:
            # -4: 強迫平倉 -3: 結構性斷裂平倉(for lag 5) -2:停損 666:正常平倉
            if position == -2 or position == -3 or position == -4 or position == 666:
                stock1_payoff = 0
                stock2_payoff = 0
            else:
                position = 0  # 剩下時間少於預期開倉時間，則不開倉，避免損失
                stock1_payoff = 0
                stock2_payoff = 0

        pos.append(position)
        stock1_profit.append(stock1_payoff)
        stock2_profit.append(stock2_payoff)
    trading_profit = sum(stock1_profit) + sum(stock2_profit)


    local_profit = trading_profit
    # local_open_num.append(trade)
    if trade == 0:  # 如果都沒有開倉，則報酬為0
        # trade_process.append([tick_data.mtimestamp.iloc[-1],"無任何交易"])
        # print("沒有開倉")
        position = 0

    if not (s1_news.empty or s2_news.empty): 
        if position == 666 and local_profit > 0:# (position == -2 or position == -3 or position == -4) and local_profit <= -10 :
            plot_spread( table.stock1[pair], table.stock2[pair], spread_all, up_open, down_open, stop_loss,
                        close, local_profit, pos, position, up_open_time, down_open_time, stop_loss_time,s1_news,s2_news)

    return local_profit, trade, trade_capital,  trading #position, tick_data[s1][0], tick_data[s2][0]      
    # table.stock1 , table.stock2 , local_profit , local_open_num , local_rt , local_std , local_skew , local_timetrend
    # #, 0


def plot_spread( stock1, stock2, spread, up_open, down_open, stop_loss, close, local_profit, pos, position,
                up_open_time, down_open_time, stop_loss_time,s1_news,s2_news):
    new_exist = 0
    plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']
    plt.figure(figsize=(20, 10))
    plt.plot(spread[:166],color= 'r')
    plt.plot(spread[165:],color ='b')
    plt.hlines(up_open, 0, len(spread) - 1, 'b')
    plt.hlines(down_open, 0, len(spread) - 1, 'b')
    plt.hlines(close + stop_loss, 0, len(spread) - 1, 'r')
    plt.hlines(close - stop_loss, 0, len(spread) - 1, 'r')
    plt.hlines(close, 0, len(spread) - 1, 'g')
    print(s1_news["9am_offset"].iloc[0],s2_news["9am_offset"].iloc[0])
    if s1_news["9am_offset"].iloc[0] >-30 and s1_news["9am_offset"].iloc[0] <180:
        plt.axvline(s1_news["9am_offset"].iloc[0]+30)
    
    if s2_news["9am_offset"].iloc[0] >-30 and s2_news["9am_offset"].iloc[0] <180:
        plt.axvline(s2_news["9am_offset"].iloc[0]+30)
        
        
    for x in range(1, len(pos)):
        if pos[x] != pos[x - 1]:
            plt.scatter(165+x, spread[165+x], color='', edgecolors='r', marker='o')
    plt.title( ' s1:' + str(stock1) + ' s2:' + str(stock2) + ' up open threshold:' + str(up_open_time) + ' down open threshold:'
              + str(down_open_time) + ' stop threshold:' + str(stop_loss_time)+'.'+str(s1_news["title"].iloc[0])+'/'+str(s2_news['title'].iloc[0]))
    if position == 666:
        plt.xlabel('profit:' + str(local_profit) + ' normal close profit')
        
    elif position == -2:
        plt.xlabel('profit:' + str(local_profit) + ' stop close profit')
    elif position == -3:
        plt.xlabel('profit:' + str(local_profit) + ' fore_lag5')
    elif position == -4:
        plt.xlabel('profit:' + str(local_profit) + ' times up，forced close profit')
    plt.savefig(path_to_image+ '_' + str(stock1) + '_' + str(stock2) + '_' + str(
        up_open_time) + '_' + str(down_open_time) + '_' + str(stop_loss_time) + '.jpg')
    plt.close('all')


if __name__ == '__main__':

    min_path = "../pair_data/{}/averageprice/"
    tick_path = "../pair_data/{}/minprice/"
    table_path = "../pair_data/newstdcompare{}/"

    # news_pr_1515 = open("./1515_news_text.txt_1_norm","a")
    # news_pr_1520 = open("./1520_news_text.txt_1_norm","a")
    # news_pr_1525 = open("./1525_news_text.txt_1_norm","a")
    news_pr_1530 = open("./1530_news_text.txt_2","a")
    # news_pr_2020 = open("./2020_news_text.txt_1_norm","a")
    # news_pr_2025 = open("./2025_news_text.txt_1_norm","a")
    # news_pr_2030 = open("./2030_news_text.txt_1_norm","a")
    # news_pr_3030 = open("./3030_news_text.txt_1_norm","a")

    for year in range(2017, 2018):
        query = "SELECT distinct(left(DateTime,10)) as td_date FROM Fintech.Stock_1Min_Price_Tick where DateTime >= '" + \
            str(year) + "-01-01 09:00' and DateTime <= '" + str(year) + "-12-31 13:30';"
        fin_cursor.execute(query)
        result = fin_cursor.fetchall()
        fin_db.commit()
        td_date = [i["td_date"] for i in result]
        for choose_date in td_date:
            print(choose_date)
            if not os.path.exists(
                table_path.format(year) +
                "{}_table.csv".format(
                    choose_date.replace(
                        "-",
                        ""))):
                continue
            tick_data = pd.read_csv(
                tick_path.format(year) +
                "{}_min_stock.csv".format(
                    choose_date.replace(
                        "-",
                        "")))[
                166:].reset_index()
            table = pd.read_csv(
                table_path.format(year) +
                "{}_table.csv".format(
                    choose_date.replace(
                        "-",
                        "")))
            for i in range(len(table)):
                is_news, local_profit, trade_capital, trade, position, s1_price, s2_price = pairs(
                    choose_date=choose_date,
                    pos=i,
                    formate_time=166,
                    table=table,
                    tick_data=tick_data,
                    maxi=5,
                    tax_cost=0.0015,
                    cost_gate=0.0030,
                    capital=300000000
                )
                print(is_news, choose_date, i, local_profit, trade_capital, trade, position, s1_price, s2_price, file=news_pr_1530)
                print(is_news, choose_date, i, local_profit, trade_capital, trade, position, s1_price, s2_price)
                # try:
                # is_news, local_profit, trade_capital, trade, position, s1_price, s2_price = pairs(
                #     choose_date=choose_date,
                #     pos=i,
                #     formate_time=166,
                #     table=table,
                #     tick_data=tick_data,
                #     maxi=5,
                #     tax_cost=0.0015,
                #     cost_gate=0.0015,
                #     capital=300000000
                # )

                # print(is_news, choose_date, i, local_profit, trade_capital, trade, position, s1_price, s2_price, file=news_pr_1515)

                # is_news, local_profit, trade_capital, trade, position, s1_price, s2_price = pairs(
                #     choose_date=choose_date,
                #     pos=i,
                #     formate_time=166,
                #     table=table,
                #     tick_data=tick_data,
                #     maxi=5,
                #     tax_cost=0.0015,
                #     cost_gate=0.0020,
                #     capital=300000000
                # )
                # print(is_news, choose_date, i, local_profit, trade_capital, trade, position, s1_price, s2_price, file=news_pr_1520)

                # is_news, local_profit, trade_capital, trade, position, s1_price, s2_price = pairs(
                #     choose_date=choose_date,
                #     pos=i,
                #     formate_time=166,
                #     table=table,
                #     tick_data=tick_data,
                #     maxi=5,
                #     tax_cost=0.0015,
                #     cost_gate=0.0025,
                #     capital=300000000
                # )
                # print(is_news, choose_date, i, local_profit, trade_capital, trade, position, s1_price, s2_price, file=news_pr_1525)




                # is_news, local_profit, trade_capital, trade, position, s1_price, s2_price = pairs(
                #     choose_date=choose_date,
                #     pos=i,
                #     formate_time=166,
                #     table=table,
                #     tick_data=tick_data,
                #     maxi=5,
                #     tax_cost=0.0020,
                #     cost_gate=0.0020,
                #     capital=300000000
                # )
                # print(is_news, choose_date, i, local_profit, trade_capital, trade, position, s1_price, s2_price, file=news_pr_2020)

                # is_news, local_profit, trade_capital, trade, position, s1_price, s2_price = pairs(
                #     choose_date=choose_date,
                #     pos=i,
                #     formate_time=166,
                #     table=table,
                #     tick_data=tick_data,
                #     maxi=5,
                #     tax_cost=0.0020,
                #     cost_gate=0.0030,
                #     capital=300000000
                # )
                # print(is_news, choose_date, i, local_profit, trade_capital, trade, position, s1_price, s2_price, file=news_pr_2030)

                # is_news, local_profit, trade_capital, trade, position, s1_price, s2_price = pairs(
                #     choose_date=choose_date,
                #     pos=i,
                #     formate_time=166,
                #     table=table,
                #     tick_data=tick_data,
                #     maxi=5,
                #     tax_cost=0.0030,
                #     cost_gate=0.0030,
                #     capital=300000000
                # )
                # print(is_news, choose_date, i, local_profit, trade_capital, trade, position, s1_price, s2_price, file=news_pr_3030)

