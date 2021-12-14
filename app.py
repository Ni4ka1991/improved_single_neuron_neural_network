#!/usr/bin/env python3


import numpy as np
from os import system
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style .use( 'seaborn-whitegrid' ) 



#EXPERIMENT DATA
time_m = [ 0, 5,  10,  20,  60 ]      # boiler running time (min)
temp_c = [ 0, 15, 20,  23,  25 ]      #temp in the house (C) max = 100C

#plt.plot( time_m, temp_c, color = "green", linestyle="solid", linewidth = 1, marker = "x" )
#plt.show()
#
#1. Square root function  График функции корень из x => 
#Частный случай общей функции. y = sqrt(x - a) + b. a смещает график по оси x, так как под знаком корня должно быть положительное число; b - смещает график по y
#2. max y = [0, 100]; x=[0, 10_000]
#3. 

system( "clear" )
#y = np.sqrt(x)
#y = w * np.sqrt(x) + b
Y = []
x = [ 5, 15,  20,  25,  30,  35  ]
w = [ 0, 0.1, 0.2, 0.3, 0.4, 0.5 ]
b = [ 0, 1,   2,   3,   4,   5   ]
for i in range(6):
#    b = np.random.normal()
#    w = np.random.normal()
    y = w[i] * np.sqrt(x[i]) + b[i]
    Y.append(y)

#print(Y)
#print(x)
plt.plot( x, Y, color = "green", linestyle="solid", linewidth = 1, marker = "x" )
plt.show()
    




