'''
Created on 27 avr. 2018

@author: fran
'''

import h5py
import os
import numpy as np

from chats_utils import load_dataset

from PIL import Image

import matplotlib.pyplot as plt

def initialize_parameters( n_x ):
    """
    Initializes parameters, shapes are:
                        W : [ n_x, 1 ]
                        b : 
    
    Returns:
    parameters -- a dictionary of tensors containing W, b
    """
    
    W = np.zeros( ( n_x, 1 ) )
    b = 0

    parameters = {"W": W,
                  "b": b,
                 }
    
    return parameters

def sigmoid( M ):

    R = np.divide( 1, np.add( 1, np.exp( - M ) ) )    
    return R

def forward_propagation( X, parameters ):

    W = parameters[ "W" ]
    b = parameters[ "b"]
    
    Z = np.dot( W.T, X ) + b
    
    A = sigmoid( Z )
    
    return A;

def backward_propagation( X, A, Y, m, learning_rate, parameters ):
    
    # get parameters
    W = parameters[ "W" ]
    b = parameters[ "b"]
    
    # Derivate by W
    dZ = A - Y
    dW = 1. / m * np.dot( X, dZ.T )
    
    # derivate by b
    db = 1. / m * np.sum( dZ )
    
    # new W and b
    W = W - learning_rate * dW
    b = b - learning_rate * db
    
    # Store result
    parameters[ "W" ] = W
    parameters[ "b" ] = b
    
    
def compute_cost( A, Y, m ):
    
    LOGPROBS = - ( np.log( A ) * Y + np.log( 1- A ) * ( 1 - Y ) )
    
    cost = 1./m * np.sum( LOGPROBS )
    return cost


def predict( X, Y, m, parameters, X_orig, errorsDir ):
    
    A = forward_propagation( X, parameters )
    
    # Transform to binary
    Y_predict = ( A >= 0.5 ).astype(int)
    
    # XOR : count errors
    errors = np.logical_xor( Y_predict, Y ).astype(int)
    nbErrors = np.sum( errors )
    
    # percentage
    exactPc = 1 - nbErrors / Y.shape[ 1 ]
    
    # Delete files in error dir
    for the_file in os.listdir( errorsDir ):
        file_path = os.path.join( errorsDir, the_file )
        try:
            if os.path.isfile(file_path):
                os.remove( file_path )
        except Exception as e:
            print(e)
        
    # Extract errors
    for i in range( 0, errors.shape[ 1 ] - 1 ): 
        # Is an error?
        if ( errors[ 0, i ] ) :
            # extract image
            X_errorImg = X_orig[ i ]
            errorImg = Image.fromarray( X_errorImg, 'RGB' )
            
            ## dump image
            errorImg.save( errorsDir + '/error-' + str( i ) + ".png", 'png' )
    
    return exactPc

def model( 
    X_train, Y_train, X_dev, Y_dev, 
    learning_rate = 0.005, nbGradientDescent = 10000,
    print_cost = True, show_plot = True):
    
    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    costs = [] # To keep track of the cost

    ## Initialize parameters
    parameters = initialize_parameters( n_x );

    # Forward propagation
    A = forward_propagation( X_train, parameters )
    cost = compute_cost( A, Y_train, m )
    
    # Run gradient descent
    for i in range( nbGradientDescent ):
        
        if ( print_cost and ( ( i % 100 ) == 0 ) ) :
            print( "Cost after iteration " + str( i ) + ": " + str( cost ) )
            
        # backward propagation
        backward_propagation( X_train, A, Y_train, m, learning_rate, parameters )
        
        # new forward prop and cost
        A = forward_propagation( X_train, parameters )
        cost = compute_cost( A, Y_train, m )
        
        if print_cost == True and ( ( i % 100 ) == 0 ):
            costs.append( cost )

            
    print( "Final cost: " + str( cost ) )
    
    # plot the cost
    plt.plot( np.squeeze( costs ) )
    plt.ylabel( 'cost' )
    plt.xlabel( 'iterations ( x100 )' )
    plt.title( "Learning rate =" + str(learning_rate) )
    if ( show_plot ) :
        plt.show()

    # Predictions
    exactTrainPc = predict( X_train, Y_train, m, parameters, X_train_orig, "C:/temp/train-errors" )
    print( "Training accuracy :", str( exactTrainPc ) )

    exactDevPc = predict( X_dev, Y_dev, m, parameters, X_dev_orig, "C:/temp/dev-errors" )
    print( "Dev accuracy      :", str( exactDevPc ) )

        
if __name__ == '__main__':

    ## Make randpm repeatable    
    np.random.seed( 1 )
    
    # Loading the dataset
    X_train_orig, Y_train_orig, X_dev_orig, Y_dev_orig = load_dataset()

    # Flatten the training and test images :de (476,64,64,3) � ( 12288, 476 ) = ( 64*64*3, 476 )
    X_train_flatten = X_train_orig.reshape( X_train_orig.shape[0], -1 ).T
    X_dev_flatten = X_dev_orig.reshape( X_dev_orig.shape[0], -1).T
    
    # Normalize image vectors
    # TODO : mean + variance normalization
    
    X_train = X_train_flatten / 255.
    X_dev   = X_dev_flatten / 255.
    
    Y_train = Y_train_orig
    Y_dev   = Y_dev_orig
    
    print ( "number of training examples = " + str( X_train.shape[1] ) )
    print ( "number of dev examples      = " + str( X_dev.shape[1] ) )
    print ( "X_train shape: " + str( X_train.shape ) )
    print ( "Y_train shape: " + str( Y_train.shape ) )
    print ( "X_test shape : " + str( X_dev.shape ) )
    print ( "Y_test shape : " + str( Y_dev.shape ) )
    
    parameters = model( 
        X_train, Y_train, X_dev, Y_dev,
        learning_rate = 0.003
        #nbGradientDescent = 1000, 
        #show_plot = False
    )

    