#!/usr/bin/env python3


import numpy as np
from os import system
import matplotlib
import matplotlib.pyplot as plt
plt.style .use( 'seaborn-whitegrid' ) 



#EXPERIMENT DATA  !!! REAL DATA !!!
time_m = [ 0, 5,  10,  20,  60 ]      # boiler running time ( min ) 
temp_c = [ 0, 15, 20,  23,  25 ]      # temp in the house (C) max = 100 C 

#plt.plot( time_m, temp_c, color = "green", linestyle="solid", linewidth = 1, marker = "x" )
#plt.show()

# GO-GO ###########################################################################################

def sigmoida( var ):
    return 1 / ( 1 + np.exp(- var ))

# NEURON 

data_quantity = len( time_m ) # lenght of real data list
w = 11
b = 0
a = 0
Y = []
S = []
ERR = []

# PREDICTED DATA !!!
def neuronFire( x ):
    global w, b, a
    y = w * np.sqrt( np.sqrt( x ))
    return y


def create_predicted_data_list( len_of_data ):
    for i in range( len_of_data ):
        Y.append( neuronFire( time_m[i] ))
    return Y

create_predicted_data_list( data_quantity )

system( "clear" )
print(Y)


def meanError( Y_real, Y_predicted ):
    pass    
   


def print_differences( list_lenght ):
    system( "clear" )
    print( "        temp_c       |   Y   |  ERROR_Y             |        ERROR_S  ")
    print("  _________________________________________________________________________")
    for i in range( list_lenght ):
        print(f"{Y[i]:20} | {temp_c[i]:5} | {ERR[i]:20} | {abs(S[i] - temp_c[i]):20}")
#############







#print_differences( data_quantity )



#plt.plot( time_m, temp_c, color = "green", linestyle="solid", linewidth = 1, marker = "x" )
#plt.plot( time_m, Y, color = "red", linestyle="solid", linewidth = 1, marker = "x" )
#plt.show()





