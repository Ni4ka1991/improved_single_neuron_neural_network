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
W_min_errors = []
#Biases = [b]
#A = [a]


system( "clear" )

# REAL-ROOT ISOLATION

k = 0 
while k < 20:
    data_table = {}                             #create an empty dictionary
    
    # w + random => calc Y_pred(w+dW) => calc ME(Y_pred) => create dict{w+dW : ME} 
    for i in range( 40 ):
        dW = np.random.normal()
        w += dW
        ME = neuron_work( w, data_quantity )     #ME - mean error
        data_table[w] = ME
    
    #convert output dictionary to np.array
    data = list( data_table.items( ))            #convert dict to list
    np_data_table = np.array( data )             #convert list to np.array
    
    
    #search w with min error in np.array
    w_min, e_min = np_data_table.min( axis = 0 )
    i, j = np.argmin(np_data_table, axis = 0 ) #индекс минимального элемента в столбце сверху-j
    w_min_error = np_data_table[ j, 0 ]
    
    #create a list of w's with min err
    W_min_errors.append( w_min_error )
    
    k += 1
#P.S. разброс значений в каждом отдельном цикле while получается существенный. От -64 до +70 

a = min( W_min_errors )
b = max( W_min_errors )
print( f"a(min) = {a} | b(max) = {b}" )




#plt.plot( Weights, Errors, color = "green", linestyle="solid", linewidth = 1, marker = "x" )
#plt.show()


















        
    














































