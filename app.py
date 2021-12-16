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
Y = []   #an empty list of predicted data
S = []
ERR = []

# PREDICTED DATA !!!
##
def neuronFire( x ):
    global w, b, a
    y = w * np.sqrt( np.sqrt( x ))
    return y
##


##
def create_predicted_data_list( len_of_data ):
    for i in range( len_of_data ):
        Y.append( neuronFire( time_m[i] ))
    return Y
##



# ERRORS PROCESSING !!! 
##
def errors_list( Y_real, Y_predicted, len_of_data ):
    for i in range( len_of_data ):
        ERR.append( abs( Y_real[i] - Y_predicted[i] ))
    return ERR
##

##
def meanError( err_list, len_of_data ):
    for i in range( len_of_data ):
        E = sum( err_list ) / len_of_data
    return E
##

##
def neuron_work( len_of_data ):
    create_predicted_data_list( len_of_data )
    errors_list( temp_c, Y, len_of_data )
    return meanError( ERR, len_of_data )
##    


# TRAIN !!!!!
num_epochs = 7
max_error = 10


Errors = [100]
Weights = [w]
#Biases = [b]
#A = [a]

for epoch in range( num_epochs ):
    print(f"##### epoch NR {epoch + 1} ########")
    dW = np.random.normal() # w >= 0
    w += dW
#    dB = np.random.normal()
#    b += dB
#    dA = np.random.normal()
#    a += dA







print( neuron_work( data_quantity ))











































