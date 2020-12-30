# -*- coding: utf-8 -*-
"""
Created on Mon May 25 18:56:37 2020

@author: allen
"""
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np
import time
import sys
import os
import pandas as pd
import math
import matplotlib.pyplot as plt
import datetime
ext_of_profit = "_profit.csv"
path_to_profit ='./profit/wavenet2018/'
path_to_image ='./profit image/'
def plot_performance_with_dd(ans,total_with_capital, dates, open_number,normal_close_number, win_rate):
    #total_with_capital = np.cumsum(total_with_capital)
    total = np.cumsum(ans)
    dd = list()
    tt =  total[0]
    r = pd.DataFrame(total_with_capital)
    #r = (total_with_capital - total_with_capital.shift(1)) / total_with_capital.shift(1)
    sharp_ratio = r.mean() / r.std() * np.sqrt(len(dates))
    for i in range(len(ans)):
        if i > 0 and total[i] > total[i-1]:
            tt = total[i]
        dd.append(total[i]-tt)
    #print(dd) 
    xs = [datetime.datetime.strptime(d, '%Y%m%d').date() for d in dates]
    highest_x = []
    highest_dt = []
    for i in range(len(total)):
        if total[i] == max(total[:i+1]) and total[i] > 0:
            highest_x.append(total[i])
            highest_dt.append(i)
    mpl.style.use('seaborn')
    f, axarr = plt.subplots(2, sharex=True, figsize=(20, 12), gridspec_kw={'height_ratios': [3, 1]})
    axarr[0].plot(np.arange(len(xs)), total, color='b', zorder=1)
    axarr[0].scatter(highest_dt, highest_x, color='lime', marker='o', s=40, zorder=2)
    axarr[0].set_title('wavenet2018', fontsize=20)
    axarr[1].bar(np.arange(len(xs)), dd, color='red')
    date_tickers = dates
    def format_date(x,pos=None):
        if x < 0 or x > len(date_tickers)-1:
            return ''
        return date_tickers[int(x)]
    axarr[0].xaxis.set_major_locator(MultipleLocator(80))
    axarr[0].xaxis.set_major_formatter(FuncFormatter(format_date))
    axarr[0].grid(True)
    shift = (max(total)-min(total))/20
    text_loc = max(total)-shift*8
    #axarr[0].text(np.arange(len(xs))[5], text_loc, 'Total open number: %d' % open_number, fontsize=15)
    axarr[0].text(np.arange(len(xs))[5], text_loc-shift, 'Total profit: %.2f' % total[-1], fontsize=15)
    axarr[0].text(np.arange(len(xs))[5], text_loc-shift*2, 'Win rate: %.2f' % (win_rate), fontsize=15)
    axarr[0].text(np.arange(len(xs))[5], text_loc-shift*5, 'sharpe ratio: %.4f' % (sharp_ratio), fontsize=15)
    #axarr[0].text(np.arange(len(xs))[5], text_loc-shift*3, 'Normal close rate: %.2f' % (normal_close_number/open_number), fontsize=15)
    axarr[0].text(np.arange(len(xs))[5], text_loc-shift*4, 'Max drawdown: %d' % min(dd), fontsize=15)
    plt.tight_layout()
    plt.savefig(path_to_image+"wavenet2018.png")
    plt.show()
    plt.close()
if __name__ =="__main__":

    datelist = [f.split('_')[0] for f in os.listdir(path_to_profit)]

    reward=[]
    cumulative_reward=[]
    capital_list=[]
    return_reward=[]
    open_num = []
    total_open = 0
    max_cap = 0
    for i,date in enumerate(sorted(datelist)):
        #print(i,date)
        profit = pd.read_csv(path_to_profit+date+ext_of_profit)
        #print(profit)
        reward.append(profit["reward"].sum())
        capital_list.append(profit["trade_capital"].sum())
        open_num.append(profit["open_num"].sum())
    print(open_num)
        
        
    max_cap = max(capital_list)
        #return_reward.append(reward[i]/capital_list[i])
    for i,date in enumerate(sorted(datelist)):
        return_reward.append(reward[i]/max_cap)

    plot_performance_with_dd(reward,return_reward,datelist,5409,3668,0.72)
    
