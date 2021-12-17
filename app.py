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
S = []

# PREDICTED DATA !!!
##
def neuronFire( w, x ):
#    global w, b, a
    y = w * np.sqrt( np.sqrt( x ))
    return y
##


##
def create_predicted_data_list( w, len_of_data ):
    Y = []   #an empty list of predicted data
    for i in range( len_of_data ):
        Y.append( neuronFire( w, time_m[i] ))
    return Y
##



# ERRORS PROCESSING !!! 
##
def errors_list( Y_real, Y_predicted, len_of_data ):
    ERR = []
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
def neuron_work( w, len_of_data ):
    Y_predicted = create_predicted_data_list( w, len_of_data )
    ERR = errors_list( temp_c, Y_predicted, len_of_data )
    return meanError( ERR, len_of_data )
#    print( f" MEAN ERROR for epoch nr.{epoch + 1} = {meanError( ERR, len_of_data )}")
##    


# TRAIN !!!!!
num_epochs = 7
max_error = 10


Errors = [100]
Weights = [w]
#Biases = [b]
#A = [a]


system( "clear" )

#random
dW = np.random.normal()
w += dW                   #add some number
    
Weights.append( w )
ME = neuron_work( w, data_quantity )     #ME - mean error
Errors.append( ME * 10 )                 # *10 - наглядней числа
    

for epoch in range( num_epochs ):
    #view
    print()
    print(f"##### epoch NR {epoch + 1} ########")
    #
    
    if Errors[-1] < Errors[-2]:
        #метод ловли льва в пустыне или Метод Больцано—Вейерштрасса
        delta_weights = abs( Weights[-1] - Weights[-2] )
        print( f"delta_weights = {delta_weights}" )
        dW = delta_weights / 2
        print( f"dW = {dW}" )
#        w += dw if Weights[-1] > Weights[-2] else w -= dw
        if Weights[-1] > Weights[-2]:
            w[epoch] += dW
        else:
            w[epoch] -= dW
    Weights.append( w[epoch] )
    print( f"w[epoch]")
    ME = neuron_work( w, data_quantity )     #ME - mean error
    Errors.append( ME * 10 )                 # *10 - наглядней числа
    
#    else:
#        print("by-by!")
    print(Errors)























        
    














































