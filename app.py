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

# GO-GO ###########################################################################################
w = 11
Y = []

for i in range( len( time_m )):
 y = w * np.sqrt( np.sqrt( time_m[i] ))
 Y.append(y)

system( "clear" )

#view data func
def print_differences( list_lenght ):
    print( "        temp_c       |   Y   |  ERROR  ")
    print("  ________ ________________________________________")
    for i in range( list_lenght ):
        print(f"{Y[i]:20} | {temp_c[i]:5} | {abs(Y[i] - temp_c[i]):20}")
#############


data_quantity = len( time_m )
print_differences( data_quantity, )



#plt.plot( time_m, temp_c, color = "green", linestyle="solid", linewidth = 1, marker = "x" )
#plt.plot( time_m, Y, color = "red", linestyle="solid", linewidth = 1, marker = "x" )
#plt.show()





