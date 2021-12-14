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
#Square root function  График функции корень из x => 
#Частный случай функции квадратного корня: y = sqrt(x - a) + b. a смещает график по оси x, так как под знаком корня должно быть положительное число (не наш случай); b - смещает график по y
#b дает значительный прирост функции по оси y, в нашем случае "скачек" по оси y наблюдается только до значения x = 20, т.е. должна быть какая-нибудь функция от b, возможно, тоже sqrt(b) 


#Для нашего случая:
#1. y = [0, 100];
#2. x = [0, 10_000]
#3. b >= 0

system( "clear" )
#y = np.sqrt(x)
#y = w * np.sqrt(x) + b
Y = []
x = [ 0,   5,   15,  20,  25,  30,  35  ]
w = [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ]
b = [ 0,   1,   2,   3,   4,   5,   6   ]
for i in range(7):
#    b = np.random.normal()
#    w = np.random.normal()
#    y = np.sqrt(x[i]) * w[i]
#    y = np.sqrt(x[i]) * w[i] + np.sqrt(b[i])
#    y = np.sqrt(x[i])
#    y = np.cbrt(x[i])
    y = np.sqrt(np.sqrt(x[i])) #the best result
    Y.append(y)

#print(Y)
#print(x)
plt.plot( x, Y, color = "green", linestyle="solid", linewidth = 1, marker = "x" )
plt.show()
    




