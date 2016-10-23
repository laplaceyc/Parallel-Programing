# -*- coding: utf-8 -*-

import os

# to get the file name list in the "AMulti" folder
list = os.listdir("AMulti")

import pandas as pd # Data visualization lib
from collections import defaultdict 

# Utilize defaultdict to store the incomming data
exec_list = defaultdict( lambda : defaultdict ( lambda : 1000 )) #   execute time
exec_value_list = defaultdict( lambda : defaultdict ( lambda : [] )) #    to sort

for file in list:
    name_list = file.split('_')
#     print name_list
    if file == '.ipynb_checkpoints': continue
    
    n = int(name_list[1][:]) //node variables
    t = int(name_list[2][3:]) // ppn variables

    with open('AMulti/'+file) as read_file:
        lines = read_file.readlines()
        exec_value_list[n][t] += [float(lines[1].strip()[19:])]
    # sort and mean the smallest 3
    exec_list[n][t] = sum(sorted(exec_value_list[n][t])[:3])/len(sorted(exec_value_list[n][t])[:3])

# 
speed_list = defaultdict( lambda : defaultdict ( lambda : 1000 ))
for n in exec_list:
    for t in exec_list[n]:
        speed_list[n][t] = exec_list[n][1]/exec_list[n][t]

# utilizing pandas to illustrate
from matplotlib import cm
test_pan = pd.Series(speed_list[4].values(), index=range(13)[1:]).sort_index()
test_pan.plot(title='Advance Data Speed Up (N = 4)',ylim=(0,12),colormap='Pastel1')
