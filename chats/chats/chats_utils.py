'''
Created on 28 avr. 2018

@author: fran
'''

import h5py
import numpy as np
import tensorflow as tf
import math
import os
import sys

from PIL import Image
from collections import OrderedDict

import matplotlib.pyplot as plt

def load_dataset( isLoadWeights ):

    # Base dir for cats and not cats images
    baseDir = os.getcwd()

    train_dataset = h5py.File( baseDir + '/data/prepared/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["y"][:]) # your train set labels

    dev_dataset = h5py.File( baseDir + '/data/prepared/dev_signs.h5', "r")
    dev_set_x_orig = np.array( dev_dataset["x"][:] ) # your test set features
    dev_set_y_orig = np.array( dev_dataset["y"][:] ) # your test set labels

    train_set_x = train_set_x_orig
    dev_set_x   = dev_set_x_orig

    # passer de (476,) (risque) a  (1,476)
    train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    dev_set_y   = dev_set_y_orig.reshape((1, dev_set_y_orig.shape[0]))

    ## replace Boolean by (1,0) values
    train_set_y = train_set_y.astype( int )
    dev_set_y   = dev_set_y.astype( int )

    # Image tags
    train_set_tag_orig = np.array(train_dataset["tag"][:]) # images tags
    dev_set_tag_orig   = np.array(dev_dataset["tag"][:]) # images tags
    # passer de (476,) (risque) a  (1,476)
    train_set_tag      = train_set_tag_orig.reshape((1, train_set_tag_orig.shape[0]))
    dev_set_tag        = dev_set_tag_orig.reshape((1, dev_set_tag_orig.shape[0]))

    # Default weight is 1 (int)
    # If weight is loaded, it is a (1,mx)
    train_set_weight = 1

    if isLoadWeights :

        ## Convert tags to weights
        train_set_weight   = getWeights( train_set_tag )

    return \
       train_set_x, train_set_y, train_set_tag, train_set_weight, \
       dev_set_x  , dev_set_y  , dev_set_tag

def getWeights( tags ):

    weights = []

    for n_tag in tags[ 0 ] :

        tag = str( n_tag )

        weight = 1
        if ( tag == "b'chats'" ) :
            weight = 100
        elif ( tag == "b'chiens'" ) :
            weight = -100
        elif ( tag == "b'loups'" ) :
            weight = -100
        elif ( tag == "b'velos'" ) :
            weight = 1
        elif ( tag == "b'gens'" ) :
            weight = 1
        elif ( tag == "b'fleurs'" ) :
            weight = 1
        elif ( tag == "b'villes'" ) :
            weight = 1
        elif ( tag == "b'voitures'" ) :
            weight = 1
        else :
            print( "Unsupported image tag", tag )
            sys.exit( 1 )

        weights.append( weight )

    # convert to numpty array
    n_weights = np.array( weights )
    # passer de (476,) (risque) a  (1,476)
    n_weights = n_weights.reshape( ( 1, n_weights.shape[0] ) )

    return n_weights

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def predict(X, parameters):

    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}

    x = tf.placeholder("float", [12288, 1])

    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})

    return prediction

def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
                                                           # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3

    return Z3

def dumpBadImages( correct, X_orig, TAG, errorsDir ):
    # Delete files in error dir
    for the_file in os.listdir( errorsDir ):
        file_path = os.path.join( errorsDir, the_file )
        try:
            if os.path.isfile(file_path):
                os.remove( file_path )
        except Exception as e:
            print(e)

    # Dico of errors by label
    mapErrorNbByTag = {}

    # Extract errors
    for i in range( 0, correct.shape[ 1 ] - 1 ):
        # Is an error?
        if ( not( correct[ 0, i ] ) ) :

            # Add nb
            label = str( TAG[ 0, i ] )
            try:
                nb = mapErrorNbByTag[ label ]
            except KeyError as e:
                ## Init nb
                nb = 0

            nb += 1
            mapErrorNbByTag[ label ] = nb

            # extract image
            X_errorImg = X_orig[ i ]
            errorImg = Image.fromarray( X_errorImg, 'RGB' )

            ## dump image
            errorImg.save( errorsDir + '/error-' + str( i ) + ".png", 'png' )

    # return dico
    return mapErrorNbByTag

def statsExtractErrors( key, X_orig, oks, TAG ) :

    errorsDir = os.getcwd().replace( "\\", "/" ) + "/errors/" + key

    os.makedirs( errorsDir, exist_ok = True )

    # Delete files in error dir
    for the_file in os.listdir( errorsDir ):

        file_path = os.path.join( errorsDir, the_file )
        try:
            if os.path.isfile(file_path):
                os.unlink( file_path )
        except Exception as e:
            print(e)

    # check deleted
    if ( len( os.listdir( errorsDir ) ) != 0 ) :
        print( "Dir", errorsDir, "not empty." )
        sys.exit( 1 )

    # Dump bad images
    mapErrorNbByTag = dumpBadImages(
        oks,
        X_orig,
        TAG,
        errorsDir
    )

    # Sort by value
    mapErrorNbByTagSorted = \
        OrderedDict( 
            sorted( mapErrorNbByTag.items(), key=lambda t: t[1], reverse=True )
    )
        
    ## Error repartition by label
    print( "Nb errors by tag for", key, ": ", mapErrorNbByTagSorted )

    ## Graph
    x = np.arange( len( mapErrorNbByTagSorted ) )
    plt.bar( x, mapErrorNbByTagSorted.values() )
    plt.xticks( x, mapErrorNbByTagSorted.keys( ) )
    plt.title( "Error repartition for " + key )
    plt.show()
