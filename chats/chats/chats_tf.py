'''
Created on 21 avr. 2018

@author: fran
'''
import math

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.python.framework import ops

from chats_utils import *

import random
import sys

# For Jupyther notebook
# %matplotlib inline

#global var X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes;

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    ### START CODE HERE ### (approx. 2 lines)
    X = tf.placeholder( tf.float32, shape=( n_x, None ), name = "X" )
    Y = tf.placeholder( tf.float32, shape=( n_y, None ), name = "Y" )
    KEEP_PROB = tf.placeholder( tf.float32, name = "KEEP_PROB" )
    ### END CODE HERE ###
    
    return X, Y, KEEP_PROB

def initialize_parameters( nbUnits, n_x ):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [1, 12]
                        b3 : [1, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    
    tf.set_random_seed(1)                   # so that your "random" numbers match ours
    
    ## Add Level 0 : X
    # example : 12228, 100, 24, 1
    nbUnits0 = [ n_x ] + nbUnits
    
    # parameters
    parameters = {}
    
    # browse layers
    for i in range( 0, len( nbUnits0 ) - 1 ):
        W_cur = tf.get_variable( "W" + str( i + 1 ), [ nbUnits0[ i + 1 ], nbUnits0[ i ] ], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b_cur = tf.get_variable( "b" + str( i + 1 ), [ nbUnits0[ i + 1 ], 1             ], initializer = tf.zeros_initializer())
        parameters[ "W" + str( i + 1 ) ] = W_cur
        parameters[ "b" + str( i + 1 ) ] = b_cur
         
    return parameters

def forward_propagation( X, parameters, nbUnits, n_x, KEEP_PROB ):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    ## Add Level 0 : X
    # example : 12228, 100, 24, 1
    nbUnits0 = [ n_x ] + nbUnits
    
    
    Z = None
    A = None
    
    curInput = X
    
    for i in range( 1, len( nbUnits0 ) ):
        # apply DropOut to hidden layer
        curInput_drop_out = curInput
        if i < len( nbUnits0 ) - 1 :
            curInput_drop_out = tf.nn.dropout( curInput, KEEP_PROB )
        
        ## Get W and b for current layer
        W_layer = parameters[ "W" + str( i ) ]
        b_layer = parameters[ "b" + str( i ) ]
        ## Linear part
        Z = tf.add( tf.matmul( W_layer, curInput_drop_out  ), b_layer )
        
        ## Activation function for hidden layers
        if i < len( nbUnits0 ) - 1 :
            A = tf.nn.relu( Z )
        
        ## Change cur input to A(layer)
        curInput = A
        
    # For last layer (output layer): no dropout and return only Z    
    return Z

def compute_cost( Z_last, Y, beta, parameters, nbUnits, n_x ):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    ## Add Level 0 : X
    # example : 12228, 100, 24, 1
    nbUnits0 = [ n_x ] + nbUnits
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose( Z_last )
    labels = tf.transpose( Y )
    
    ### START CODE HERE ### (1 line of code)
    raw_cost = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))
    
    # Loss function using L2 Regularization
    regularizer = None
    
    if ( beta != 0 ) :
        losses = []
        for i in range( 1, len( nbUnits0 ) ) :
            W_cur = parameters[ 'W' + str( i ) ]
            losses.append( tf.nn.l2_loss( W_cur ) )
            
        regularizer = tf.add_n( losses )
           
    cost = None
    
    if ( regularizer != None ) :
        cost = tf.reduce_mean( raw_cost + beta * regularizer )
    else :
        cost = tf.reduce_mean( raw_cost )
        
    return cost

def model( 
    nbUnits, X_train, Y_train, X_dev, Y_dev, 
    learning_rate = 0.0001, beta = 0, keep_prob = [1,1,1],
    num_epochs = 1900, minibatch_size = 32, 
    print_cost = True, show_plot = True, extractImageErrors = True
    ):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    ### START CODE HERE ### (1 line)
    X, Y, KEEP_PROB = create_placeholders( n_x, n_y )
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters( nbUnits, n_x )
    ### END CODE HERE ###
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z_last = forward_propagation( X, parameters, nbUnits, n_x, KEEP_PROB )
    ### END CODE HERE ###
    
    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost( Z_last, Y, beta, parameters, nbUnits, n_x )
    ### END CODE HERE ###
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer( learning_rate ).minimize( cost )
    ### END CODE HERE ###
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , minibatch_cost = sess.run( [optimizer,cost], feed_dict={ X: minibatch_X, Y: minibatch_Y, KEEP_PROB: keep_prob } )
                ### END CODE HERE ###
                
                # print( "Minibatch 0 cost:",  minibatch_cost )
                # sys.exit( "bye" )
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        if ( show_plot ) :
            plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        prediction = tf.round( tf.sigmoid( Z_last ) )
        correct_prediction = tf.equal( prediction, Y )

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        accuracyTrain = accuracy.eval({X: X_train, Y: Y_train, KEEP_PROB: 1 } )
        print ( "Train Accuracy:", accuracyTrain )
        
        accuracyDev = accuracy.eval({X: X_dev, Y: Y_dev, KEEP_PROB: 1 } )
        print ( "Dev Accuracy:", accuracyDev )

        if ( extractImageErrors ) :
            
            # Dump bad images
            dumpBadImages( 
                correct_prediction.eval( {X: X_train, Y: Y_train, KEEP_PROB: 1 } ),
                X_train_orig,
                "C:/temp/train-tf-errors"
            )
            dumpBadImages( 
                correct_prediction.eval( {X: X_dev, Y: Y_dev, KEEP_PROB: 1 } ),
                X_dev_orig,
                "C:/temp/dev-tf-errors"
            )
        
        # Serialize parameters        
        paramsFileName = "saved/params-Beta" + str( beta ) + "-keepProb"  + str( keep_prob )+ ".bin"
        print( "Serialize parameters to " + paramsFileName )
        
        with open( paramsFileName, "wb" ) as fpOut:
            for key, value in parameters.items():
                # Key
                bKey = bytearray( key, "UTF-8" )
                fpOut.write( bKey )
                # Array
                np.save( fpOut, value )
    
    
        return parameters, accuracyDev, accuracyTrain

def model_batch( 
    nbUnits, X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, 
    beta = 0, keep_prob = [1,1,1],
    num_epochs = 1900, minibatch_size = 32):
    
    return model( 
        nbUnits, X_train, Y_train, X_test, Y_test, learning_rate, 
        beta, keep_prob,
        num_epochs, minibatch_size, print_cost = True, show_plot = False, extractImageErrors = False
    )
    
def tuning( num_epochs, learning_rate ):
    beta_min = 0.000000000000001
    beta_max = 0.5
    
    keep_prob_min = 0.5
    keep_prob_max = 1
    
    # store results
    tuning = {}
    
    # max accuracyDev
    maxAccuracyDev = -1;
    
    maxHyperParams = {}
    
    nbTuning = 20
    
    for j in range( 1, nbTuning ) :
        
        print( "*****************************" )
        print( "Tune round", str( j ), "/", str( nbTuning ) )
        print( "*****************************" )
        
        # calculate beta
        logBeta = random.uniform( math.log10( beta_min ), math.log10( beta_max ) )
        beta = math.pow( 10, logBeta )
        print( "Beta = " + str( beta ))
        
        # calculate keep_prob
        logKeep_prob = random.uniform( math.log10( keep_prob_min ), math.log10( keep_prob_max ) )
        keep_prob = math.pow( 10, logKeep_prob )
        print( "keep_prob = " + str( keep_prob ))
        
        _, accuracyDev, accuracyTrain = model_batch( 
            nbUnits, X_train, Y_train, X_dev, Y_dev, 
            beta = beta, keep_prob = keep_prob,
            num_epochs = num_epochs, learning_rate = learning_rate
        )
    
        # Store results
        tuning[ j ] = { 
            "beta": beta, "keep_prob": keep_prob, 
            "accuracyDev": accuracyDev, "accuracyTrain": accuracyTrain
        }
    
        # Max
        if ( accuracyDev > maxAccuracyDev ) :
            maxAccuracyDev = accuracyDev
            maxHyperParams = tuning[ j ]
            
        # print max
        print( "Max DEV accuracy:", maxAccuracyDev )
        print( "Max hyper params:" )
        print( maxHyperParams )
        
            
    # Print tuning
    print( "Tuning:" )
    print( tuning )
    
    print( "Max hyper params:" )
    print( maxHyperParams )


if __name__ == '__main__':
    
    print( "***************************************************************" )
    print( "Cat's recognition with TensorFlow - using parameterized network" )
    print( "***************************************************************" )

    # Make sure random is predictible...
    np.random.seed( 1 )
    
    ## Init tensorflow multi-threading
    # When TF 1.8 available...
#     config = tf.ConfigProto()
#     config.intra_op_parallelism_threads = 16
#     config.inter_op_parallelism_threads = 16
#     tf.session(config=config)
    
    ## Units of layers
    nbUnits = [ 50, 24, 12, 1 ]
    learning_rate = 0.0001
    num_epochs = 1000
    # Result from tuning
    beta = 0
    keep_prob = 1
    learning_rate = 0.0001
    num_epochs = 1000
    
    #nbUnits = [ 100, 48, 1 ]
    # Result from tuning
    #beta = 1.6980624617370184e-15
    #keep_prob = 0.724123179663981
    
#     nbUnits = [ 25, 12, 1 ]
#     # Result from tuning
#     beta = 6.531654400821318e-14
#     keep_prob = 0.8213956561201344
#     learning_rate = 0.0001
#     num_epochs = 1500
  
    # Loading the dataset
    X_train_orig, Y_train_orig, X_dev_orig, Y_dev_orig = load_dataset()

    # Flatten the training and test images
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_dev_flatten = X_dev_orig.reshape(X_dev_orig.shape[0], -1).T
    # Normalize image vectors
    X_train = X_train_flatten/255.
    X_dev = X_dev_flatten/255.
    
    Y_train = Y_train_orig
    Y_dev = Y_dev_orig

    print ("number of training examples = " + str(X_train.shape[1]))
    print ("number of test examples = " + str(X_dev.shape[1]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_dev.shape))
    print ("Y_test shape: " + str(Y_dev.shape))
    
    # Run model
    print( "Units:")
    print( nbUnits )
    
    tuning( num_epochs = num_epochs, learning_rate = learning_rate )
    
#     model( 
#         nbUnits, X_train, Y_train, X_dev, Y_dev, 
#         beta = beta, keep_prob = keep_prob,
#         num_epochs = num_epochs, learning_rate = learning_rate
#     )
    