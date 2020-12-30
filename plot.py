# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 21:12:24 2020

@author: allen
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager
 
a = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
from matplotlib.font_manager import _rebuild

_rebuild()
for i in a:
    print(i)
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']

sales = [100,40,54]
x_labels =['A品牌','B品牌','C品牌']
plt.bar(x_labels,sales)
plt.show()
