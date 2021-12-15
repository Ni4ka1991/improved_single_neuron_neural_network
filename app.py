#!/usr/bin/env python3


import numpy as np
from os import system
import matplotlib
import matplotlib.pyplot as plt
plt.style .use( 'seaborn-whitegrid' ) 



#EXPERIMENT DATA
time_m = [ 0, 5,  10,  20,  60 ]      # boiler running time (min)
temp_c = [ 0, 15, 20,  23,  25 ]      # temp in the house (C) max = 100 C

#plt.plot( time_m, temp_c, color = "green", linestyle="solid", linewidth = 1, marker = "x" )
#plt.show()



#CLASSIC SOLUTION #################################################################################
#Square root function  График функции корень из x => 
#Частный случай функции квадратного корня: y = sqrt(x - a) + b. a смещает график по оси x, так как под знаком корня должно быть положительное число (не наш случай); b - смещает график по y
#b дает значительный прирост функции по оси y, в нашем случае "скачек" по оси y наблюдается только до значения x = 20, т.е. должна быть какая-нибудь функция от b, возможно, тоже sqrt(b) 


#Для нашего случая:
#1. y = [0, 100];
#2. x = [0, 10_000]
#3. b >= 0


#y = w * np.sqrt( np.sqrt(x[i] ))

# ##################################################################################################


# TASK #############################################################################################

# I'm comfortable with temperature of 26.5 in the house.
#Task: How long should the boiler be on?

#Questions raised:
#1. Is the house cold? Should I take into accaunt the initial temperature of the house?

###################################################################################################

# GO-GO ###########################################################################################

w = 11
b = 0
Y = []

for i in range( len( time_m )):
 y = w * np.sqrt( np.sqrt( time_m[i] )) + b
 Y.append(y)

system( "clear" )

plt.plot( time_m, temp_c, color = "green", linestyle="solid", linewidth = 1, marker = "x" )
plt.plot( time_m, Y, color = "red", linestyle="solid", linewidth = 1, marker = "x" )
plt.show()





