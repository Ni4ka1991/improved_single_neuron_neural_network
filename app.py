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

# NEURON 

data_quantity = len( time_m ) # lenght of real data list
w = 11
b = 0
a = 0
S = []

# PREDICTED DATA !!!
##
def neuronFire( w, x ):
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
##    


# TRAIN !!!!!
num_epochs = 10
max_error = 1.3
W_min_errors = []

#Biases = [b]
#A = [a]


system( "clear" )

## REAL-ROOT ISOLATION


###1. create a one table of n-random weights values: w + random => calc Y_pred(w+dW) => calc ME(Y_pred) => create dict{w+dW : ME} 
###2. select ONE minim error and corresponding value of weight

def random_min_error( n, weight, len_of_data ):
    data_table = {}                             #create an empty dictionary
    
    for i in range( n ):
        dW = np.random.normal()
        weight += dW
        ME = neuron_work( weight, len_of_data )     #ME - mean error
        data_table[weight] = ME

    ###convert output dictionary to np.array
    data = list( data_table.items( ))            #convert dict to list
    np_data_table = np.array( data )             #convert list to np.array
    
    ###search w with min error in np.array
    w_min, e_min = np_data_table.min( axis = 0 )
    i, j = np.argmin(np_data_table, axis = 0 ) #индекс минимального элемента в столбце сверху-j
    w_min_error = np_data_table[ j, 0 ]
    return w_min_error


### out a LIST of n-w_minim errors values 
def list_w_min_errors( n, weight, len_of_data ):
    W_min_errors = []
    for i in range( n ):
        w_min_err = random_min_error( 20, weight, len_of_data )
        W_min_errors.append( w_min_err )
    return W_min_errors

W_min_errors = list_w_min_errors( 20, w, data_quantity )


### search w_min and w_max in W_min_errors list REAL-ROOT ISOLATION POINTS!!!
w_min = min( W_min_errors )
w_max = max( W_min_errors )

#P.S. разброс значений в каждом отдельном цикле while получается существенный. От -64 до +70 




## APPLICATION HALF-DIVISION METHOD

### create a tuple of ME in max, min
def me_critical_points_calculation( var_min, var_max, len_of_data ):
    
    me_critical_points = []
    
    me_w_min  = neuron_work( var_min,   len_of_data ) 
    me_critical_points.append( me_w_min )
    
    me_w_max  = neuron_work( var_max,   len_of_data ) 
    me_critical_points.append( me_w_max )
    
    return tuple( me_critical_points )


###create an INITIAL tuple of ME in max, min and half points in Real-Root Isolation Points  
ME_critical_points = me_critical_points_calculation( w_min, w_max, data_quantity )                 #tuple 
ME_critical_points_list = list( me_critical_points_calculation( w_min, w_max, data_quantity ))     #list for using in loop

###
### start data
w_max  = ME_critical_points[-1]
w_min  = ME_critical_points[0]

for i in range( num_epochs ):
    
    print( "\n" + "#" * 12 )
    print( f"Epoch NR {i + 1}" )
    
    if( w_min <= max_error ):
        print( f"\nW_supper_hero (me_w_min) = {ME_critical_points[0] :20 }\n" )
        break
    elif( w_max <= max_error ):
        print( f"\nW_supper_hero (me_w_max) = {ME_critical_points[-1] :20}\n" )
        break
    
    else:
        w_half = ( w_max + w_min ) / 2
        w_max = w_half        
        print( f" w_min = {w_min:20} : w_max = {w_max:20} : w_half = {w_half:20}" )       



        
    














































