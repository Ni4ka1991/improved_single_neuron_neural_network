#!/usr/bin/env python3


import numpy as np
import math
from os import system
import matplotlib
import matplotlib.pyplot as plt
plt.style .use( 'seaborn-whitegrid' ) 



#EXPERIMENT DATA  !!! REAL DATA !!!
#time_m = [ 0, 5,  10,  20,  60 ]      # boiler running time ( min ) 
#temp_c = [ 0, 15, 20,  23,  25 ]      # temp in the house (C) max = 100 C 

time_m = [ 5,  10,  20,  60 ]      # boiler running time ( min ) 
temp_c = [ 15, 20,  23,  25 ]      # temp in the house (C) max = 100 C 

temp_hw = [ 15, 16.5, 18.5,  20,   22,  23,  23.5, 24,  25 ]

##view experimental data
#print(temp_c)
#plt.plot( time_m, temp_c, color = "green", linestyle="solid", linewidth = 1, marker = "x" )
#plt.show()

# GO-GO ###########################################################################################

# NEURON 

data_quantity = len( time_m ) # lenght of real data list
w = 25
b = 7
x = 5


# PREDICTED DATA !!!
##
def neuronFire( w, x, b ):
    y = w * ( 1 - math.pow( math.e, ( -1 * x / b ))) # x = time_m, w,b = coeff
    return y
##


## MY neuron calculate result for experimental data by MY func
def create_predicted_data_list( w, x, b, len_of_data ):
    Y = []   #an empty list of predicted data
    for i in range( len_of_data ):
        Y.append( neuronFire( w, time_m[i], b ))
    return Y
##

##view predicted data
#Y_pred = create_predicted_data_list( w, time_m, b, data_quantity )
#print( Y_pred )
#plt.plot( time_m, Y_pred, color = "red", linestyle="solid", linewidth = 1, marker = "x" )
#plt.show()
##


# ERRORS PROCESSING !!! 
##

##модуль разности данных в каждой отдельно взятой точке
def errors_list( Y_real, Y_predicted, len_of_data ):
    ERR = []
    for i in range( len_of_data ):
        ERR.append( abs( Y_real[i] - Y_predicted[i] ))
    return ERR
##
##view mear error in each point
#point_mean_error = errors_list( temp_c, Y_pred, data_quantity )
#print( point_mean_error )

##
##calculation average general error
def meanError( err_list, len_of_data ):
    for i in range( len_of_data ):
        E = sum( err_list ) / len_of_data
    return E
##

##
def neuron_work( w, x, b, len_of_data ):
    Y_predicted = create_predicted_data_list( w, x, b, len_of_data )
    ERR = errors_list( temp_c, Y_predicted, len_of_data )
    return meanError( ERR, len_of_data )
##    

#neuron_err = neuron_work( w, x, b, data_quantity )
#print(neuron_err)
#input( "hit...  " )


# TRAIN !!!!!
num_epochs = 10_000
max_error = 0.2
W_min_errors = []
B_min_errors = []



system( "clear" )

## REAL-ROOT ISOLATION


###1. create a one table of n-random weights values: w + random => calc Y_pred(w+dW) => calc ME(Y_pred) => create dict{w+dW : ME} 
###2. select ONE minim error and corresponding value of weight

def random_min_error( n, w, x, b, len_of_data ):
    
    data_table = {}                             #create an empty dictionary
    
    for i in range( n ):
        key_data_table = []
        
        dW = np.random.normal()
        dB = np.random.normal()
        
        w += dW
        key_data_table.append( w )
        b += dB
        key_data_table.append( b )
        
        print( f"A {n} pair of random values: {key_data_table}" )
        key_data_table = tuple( key_data_table )
        
        ME = neuron_work( w, x, b, len_of_data )     #ME - mean error

        data_table[ key_data_table ] = ME
    
    print("#")
    print( data_table )

    ###convert output dictionary to np.array
    data = list( data_table.items( ))            #convert dict to list
    np_data_table = np.array( data, dtype = 'object' )             #convert list to np.array
    
    ###search w with min error in np.array
    w_b_min, e_min = np_data_table.min( axis = 0 )
    print("#")
    print( f"Val of min error == {e_min}" )
    i, j = np.argmin(np_data_table, axis = 0 ) #индекс минимального элемента в столбце сверху-j
    w_b_min_error = np_data_table[ j, 0 ]
    input( "hit ..." )
    return w_b_min_error

##view table of errors in func of random values w, b 
#a = random_min_error( 40, w, time_m, b, data_quantity )
#print(f"Values with min error: w => - {a} <= b ")
#input( "hit ..." )

### out a LIST of n-w_minim errors values 
def list_w_b_min_errors( n, w, x, b, len_of_data ):
    W_B_min_errors = []
    for i in range( n ):
        w_b_min_err = random_min_error( n, w, x, b, len_of_data )
        W_B_min_errors.append( w_b_min_err )
    return W_B_min_errors

### out a list of 33 weights with min errors
W_B_min_errors = list_w_b_min_errors( 33, w, time_m, b, data_quantity )
print( "List of values with minim values: {W_B_min_errors}" )


### search w_min and w_max in W_min_errors list REAL-ROOT ISOLATION POINTS!!!
w_min = min( W_min_errors )
w_max = max( W_min_errors )

# --- helpfull view
#print(w_min)
#print(w_max)
#input("hit")
# ----------------
#P.S. разброс значений в каждом отдельном цикле while получается существенный. От -64 до +70 




## APPLICATION HALF-DIVISION METHOD


for i in range( num_epochs ):
    
    me_w_min  = neuron_work( w_min,   data_quantity ) 
    me_w_max  = neuron_work( w_max,   data_quantity ) 
    
    if( me_w_min <= max_error ):
        print( f"\nW_supper_hero (me_w_min) = {ME_critical_points[0] :20 }\n" )
        break
    
    elif( me_w_max <= max_error ):
        print( f"\nW_supper_hero (me_w_max) = {ME_critical_points[-1] :20}\n" )
        break
    
    elif( me_w_max == me_w_min ):
        print( "Nice to see you!!!" )
        break
    
    else:
        w_half = ( w_max + w_min ) / 2
        w_max = w_half        

#### END ####

#view data


time_hw = []

for i in range( len( temp_hw ) ):
    time = np.square(np.square( temp_hw[i]/w ))
    time = time.round( 1 )
    time_hw.append( time )


plt.plot( time_hw, temp_hw, color = "red", linestyle="solid", linewidth = 1, marker = "x" )
plt.plot( time_m, temp_c, color = "green", linestyle="solid", linewidth = 1, marker = "x" )
plt.show()













































