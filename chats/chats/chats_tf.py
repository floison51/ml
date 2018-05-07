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

import const.constants as const
import db.db as db

import random
import sys

import time

# For Jupyther notebook
# %matplotlib inline

# Tensorboard log dir
APP_KEY = "chats"
TENSORBOARD_LOG_DIR = os.getcwd().replace( "\\", "/" ) + "/run/tf-board/" + APP_KEY
DB_DIR              = os.getcwd().replace( "\\", "/" ) + "/run/db/" + APP_KEY

def variable_summaries( var ):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


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

def initialize_parameters( structure, n_x, isTensorboardFull ):
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
    structure0 = [ n_x ] + structure

    # parameters
    parameters = {}

    # browse layers
    for i in range( 0, len( structure0 ) - 1 ):

        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope( "Layer" + str( i+1 ) ):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope( 'weights' ):
                W_cur = tf.get_variable( "W" + str( i + 1 ), [ structure0[ i + 1 ], structure0[ i ] ], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
                if isTensorboardFull :
                    variable_summaries( W_cur )

            with tf.name_scope( 'bias' ):
                b_cur = tf.get_variable( "b" + str( i + 1 ), [ structure0[ i + 1 ], 1             ], initializer = tf.zeros_initializer())
                if isTensorboardFull :
                    variable_summaries( b_cur )

        parameters[ "W" + str( i + 1 ) ] = W_cur
        parameters[ "b" + str( i + 1 ) ] = b_cur

    return parameters

def forward_propagation( X, parameters, structure, n_x, KEEP_PROB, isTensorboardFull ):
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
    structure0 = [ n_x ] + structure


    Z = None
    A = None

    curInput = X

    for i in range( 1, len( structure0 ) ):
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope( "Layer" + str( i ) ):
            # apply DropOut to hidden layer
            curInput_drop_out = curInput
            if i < len( structure0 ) - 1 :
                curInput_drop_out = tf.nn.dropout( curInput, KEEP_PROB )

            ## Get W and b for current layer
            W_layer = parameters[ "W" + str( i ) ]
            b_layer = parameters[ "b" + str( i ) ]

            ## Linear part
            with tf.name_scope( 'Z' ):
                Z = tf.add( tf.matmul( W_layer, curInput_drop_out  ), b_layer )
                if isTensorboardFull :
                    tf.summary.histogram( 'Z', Z )

            ## Activation function for hidden layers
            if i < len( structure0 ) - 1 :
                A = tf.nn.relu( Z )
                if isTensorboardFull :
                    tf.summary.histogram( 'A', A   )

            ## Change cur input to A(layer)
            curInput = A

    # For last layer (output layer): no dropout and return only Z
    return Z

def compute_cost( Z_last, Y, WEIGHT, beta, parameters, structure, n_x ):
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
    structure0 = [ n_x ] + structure

    with tf.name_scope('cross_entropy'):

        # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
        logits = tf.transpose( Z_last )
        labels = tf.transpose( Y )

        raw_cost = None

        with tf.name_scope( 'total' ):
            if ( ( type( WEIGHT ) == int ) and ( WEIGHT == 1 ) ) :
                raw_cost = \
                    tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits( logits = logits, labels = labels )
                    )

            else :
                # Use image weights to reduce false positives
                # pos_weight = tf.transpose( WEIGHT )
                raw_cost = \
                    tf.reduce_mean(
                        tf.nn.weighted_cross_entropy_with_logits(logits = logits, targets = labels, pos_weight = WEIGHT )
                    )

            tf.summary.scalar( 'raw_cost', raw_cost)

            # Loss function using L2 Regularization
            regularizer = None

            if ( beta != 0 ) :
                losses = []
                for i in range( 1, len( structure0 ) ) :
                    W_cur = parameters[ 'W' + str( i ) ]
                    losses.append( tf.nn.l2_loss( W_cur ) )

                regularizer = tf.add_n( losses )

            cost = None

            if ( regularizer != None ) :
                cost = tf.reduce_mean( raw_cost + beta * regularizer )
            else :
                cost = raw_cost

            tf.summary.scalar( 'cost', cost)

    return cost

def runIteration(
    iteration, num_minibatches, sess, feed_dict,
    optimizer, cost,
    train_writer, dev_writer, mergedSummaries,
    isTensorboard
):

    # No mini-batch
    if ( isTensorboard and ( iteration % ( 100 * num_minibatches ) == 100 * num_minibatches - 1 ) ):  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ , curCost = sess.run(
            [mergedSummaries,optimizer,cost], feed_dict=feed_dict,
            options=run_options,run_metadata=run_metadata
        )
        train_writer.add_run_metadata( run_metadata, 'step%03d' % iteration )
        train_writer.add_summary( summary, iteration )
    else :

        if ( isTensorboard ) :
            #run without meta data
            summary, _ , curCost = sess.run(
                [mergedSummaries,optimizer,cost], feed_dict=feed_dict
            )
            train_writer.add_summary( summary, iteration )
        else :
            _ , epoch_cost = sess.run(
                [optimizer,cost], feed_dict=feed_dict
            )

    return curCost

def model(
    conn, idRun, structure,
    X_train, Y_train, PATH_train, TAG_train, WEIGHT_train,
    X_dev, Y_dev, PATH_dev, TAG_dev,
    hyperParams,
    print_cost = True, show_plot = True, extractImageErrors = True,
    isTensorboard = True, isTensorboardFull = False
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

    # Get hyper parameters from dico
    beta        = hyperParams[ const.KEY_BETA ]
    keep_prob   = hyperParams[ const.KEY_KEEP_PROB ]
    num_epochs  = hyperParams[ const.KEY_NUM_EPOCHS ]
    minibatch_size      = hyperParams[ const.KEY_MINIBATCH_SIZE ]
    start_learning_rate = hyperParams[ const.KEY_START_LEARNING_RATE ]

    # Create Placeholders of shape (n_x, n_y)
    ### START CODE HERE ### (1 line)
    X, Y, KEEP_PROB = create_placeholders( n_x, n_y )
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters( structure, n_x, isTensorboardFull = isTensorboardFull )
    ### END CODE HERE ###

    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z_last = forward_propagation( X, parameters, structure, n_x, KEEP_PROB, isTensorboardFull = isTensorboardFull )
    ### END CODE HERE ###

    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost( Z_last, Y, WEIGHT_train, beta, parameters, structure, n_x )
    ### END CODE HERE ###

    # Back-propagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    # Decaying learning rate
    global_step = tf.Variable( 0 )  # count the number of steps taken.
    with tf.name_scope('cross_entropy'):
        with tf.name_scope( 'total' ):
            learning_rate = tf.train.exponential_decay(
                start_learning_rate, global_step, 10000, 0.96, staircase=True
            )
            tf.summary.scalar( 'learning_rate', learning_rate )

    # fixed learning rate
#     learning_rate = start_learning_rate

    # Adam optimizer
    optimizer = tf.train.AdamOptimizer( learning_rate ).minimize( cost, global_step = global_step )

    with tf.name_scope('accuracy'):
        # To calculate the correct predictions
        prediction = tf.round( tf.sigmoid( Z_last ) )
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal( prediction, Y )
        with tf.name_scope('accuracy'):
            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    mergedSummaries = tf.summary.merge_all()

    # Start time
    tsStart = time.time()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        train_writer = tf.summary.FileWriter( TENSORBOARD_LOG_DIR + '/train', sess.graph )
        dev_writer   = tf.summary.FileWriter( TENSORBOARD_LOG_DIR + '/dev', sess.graph )

        # Run the initialization
        init = tf.global_variables_initializer()
        sess.run( init )

        # current iteration
        iteration = 0

        ## optimisation may overshoot locally
        ## To avoid returning an overshoot, we detect it and run extra epochs if needed
        finalizationMode = False
        current_num_epochs = num_epochs
        iEpoch = 0
        minCost = 99999999999999
        minCostFinalization = 99999999999999
        finished = False

        # Do the training loop
        while ( not finished and ( iEpoch <= current_num_epochs ) ) :

            epoch_cost = 0.                       # Defines a cost related to an epoch

            if ( minibatch_size < 0 ) :

                # No mini-batch : do a gradient descent for whole data
                epoch_cost = runIteration(
                    epoch, 1, sess,
                    { X: X_train, Y: Y_train, KEEP_PROB: keep_prob },
                    optimizer, cost,
                    train_writer, dev_writer, mergedSummaries,
                    isTensorboard
                )

                iteration += 1

            else:
                #Minibatch mode
                num_minibatches = int( m / minibatch_size ) # number of minibatches of size minibatch_size in the train set
                seed = seed + 1
                minibatches = random_mini_batches( X_train, Y_train, minibatch_size, seed )

                for minibatch in minibatches:

                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch

                    minibatch_cost = runIteration(
                        iteration, num_minibatches, sess,
                        { X: minibatch_X, Y: minibatch_Y, KEEP_PROB: keep_prob },
                        optimizer, cost,
                        train_writer, dev_writer, mergedSummaries,
                        isTensorboard
                    )

                    epoch_cost += minibatch_cost / num_minibatches
                    iteration += 1

            if print_cost == True and iEpoch % 100 == 0:
                print ("Cost after epoch %i, iteration %i: %f" % ( iEpoch, iteration, epoch_cost ) )

            if print_cost == True and iEpoch % 5 == 0:
                costs.append( epoch_cost )

            # Record min cost
            minCost = min( minCost, epoch_cost )

            # Next epoch
            iEpoch += 1

            # Close to finish?
            if ( not finalizationMode and ( iEpoch > current_num_epochs ) ) :
                # Activate finalization mode
                finalizationMode = True
                # local overshoot?
                if ( epoch_cost > minCost ) :
                    # Yes, run some extra epochs
                    print( "WARNING: local cost overshoot detected, adding maximum 100 epochs to leave local cost overshoot" )
                    current_num_epochs += 100
                    minCostFinalization = minCost

            if ( finalizationMode ) :
                # Check overshoot is finished
                if ( epoch_cost <= minCostFinalization ) :
                    # finished
                    finished = True

        # Close tensorboard streams
        train_writer.close()
        dev_writer.close()

        # Final cost
        print( "Final cost:", epoch_cost )

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Start learning rate =" + str( start_learning_rate ) )
        if ( show_plot ) :
            plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        accuracyTrain = accuracy.eval({X: X_train, Y: Y_train, KEEP_PROB: 1 } )
        print ( "Train Accuracy:", accuracyTrain )

        accuracyDev = accuracy.eval({X: X_dev, Y: Y_dev, KEEP_PROB: 1 } )
        print ( "Dev Accuracy:", accuracyDev )

        ## Errors
        resultInfo = {}

        if ( extractImageErrors ) :

            # Lists of OK for training
            oks_train = correct_prediction.eval( {X: X_train, Y: Y_train, KEEP_PROB: 1 } )
            map1, map2 = statsExtractErrors( "train", X_orig = X_train_orig, oks = oks_train, PATH = PATH_train, TAG = TAG_train, show_plot=show_plot )
            # Errors nb by data tag
            resultInfo[ const.KEY_TRN_NB_ERROR_BY_TAG ] = map1
            resultInfo[ const.KEY_TRN_PC_ERROR_BY_TAG ] = map1

            oks_dev = correct_prediction.eval( {X: X_dev, Y: Y_dev, KEEP_PROB: 1 } )
            map1, map2 = statsExtractErrors( "dev", X_orig= X_dev_orig, oks = oks_dev, PATH = PATH_dev, TAG = TAG_dev, show_plot=show_plot )
            # Errors nb by data tag
            resultInfo[ const.KEY_DEV_NB_ERROR_BY_TAG ] = map1
            resultInfo[ const.KEY_DEV_PC_ERROR_BY_TAG ] = map1

        # Serialize parameters
#         paramsFileName = "saved/params-Beta" + str( beta ) + "-keepProb"  + str( keep_prob )+ ".bin"
#         print( "Serialize parameters to " + paramsFileName )
#
#         with open( paramsFileName, "wb" ) as fpOut:
#             for key, value in parameters.items():
#                 # Key
#                 bKey = bytearray( key, "UTF-8" )
#                 fpOut.write( bKey )
#                 # Array
#                 np.save( fpOut, value )

    # End time
    tsEnd = time.time()

    ## Elapsed (seconds)
    elapsedSeconds = tsEnd - tsStart
    # performance index : per iEpoth - per samples
    perfIndex = elapsedSeconds / iEpoch / n_x * 1e6

    perfInfo = {
        const.KEY_ELAPSED_SECOND: elapsedSeconds,
    }

    print( "Elapsed (s):", elapsedSeconds )
    print( "Perf index :", perfIndex )
    
    # Update DB run after execution, add extra info
    db.updateRunAfter(
        conn, idRun,
        perf_info = perfInfo, result_info=resultInfo,
        perf_index=perfIndex, 
        train_accuracy=accuracyTrain.astype( float ), 
        dev_accuracy=accuracyDev.astype( float )
    )

    return parameters, accuracyDev, accuracyTrain

def model_batch(
    structure, X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
    beta = 0, keep_prob = [1,1,1],
    num_epochs = 1900, minibatch_size = 32):

    return model(
        structure, X_train, Y_train, X_test, Y_test, learning_rate,
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
            structure, X_train, Y_train, X_dev, Y_dev,
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

    print( "TensorFlow version:", tf.__version__ )
    # Tensorboard output dir
    print( "TensorBoard dir:", TENSORBOARD_LOG_DIR )
    print( "Db dir         :", DB_DIR )

    # clean TF log dir
    if tf.gfile.Exists( TENSORBOARD_LOG_DIR ):
        tf.gfile.DeleteRecursively( TENSORBOARD_LOG_DIR )
        tf.gfile.MakeDirs( TENSORBOARD_LOG_DIR )

    # Make sure random is predictible...
    np.random.seed( 1 )

    # system info
    systemInfo = getSystemInfo( tf.__version__ )

    # hyper parameters
    hyperParams = {}

    ## Init tensorflow multi-threading
    # When TF 1.8 available...
#     config = tf.ConfigProto()
#     config.intra_op_parallelism_threads = 16
#     config.inter_op_parallelism_threads = 16
#     tf.session(config=config)

#     ## Units of layers
    structure                                   = [ 1 ]
    hyperParams[ const.KEY_MINIBATCH_SIZE ]     = 64
    hyperParams[ const.KEY_NUM_EPOCHS ]         = 20
    hyperParams[ const.KEY_USE_WEIGHTS ]        = False
    hyperParams[ const.KEY_START_LEARNING_RATE ]= 0.003
    hyperParams[ const.KEY_BETA ]               = 0
    hyperParams[ const.KEY_KEEP_PROB ]          = 1

    ## Units of layers
#     hyperParams[ const.KEY_NB_UNITS ] = [ 100, 48, 1 ]
#     # Mini-batch
#     hyperParams[ const.KEY_MINIBATCH_SIZE ] = 64
#     hyperParams[ const.KEY_NUM_EPOCHS ] = 2000
#     hyperParams[ const.KEY_USE_WEIGHTS ] = False
#     hyperParams[ const.KEY_LEARNING_RATE ] = 0.0001
#     hyperParams[ const.KEY_BETA ] = 0 #1.6980624617370184e-15
#     hyperParams[ const.KEY_KEEP_PROB ] = 0.724123179663981
    # {'beta': 1.6980624617370184e-15, 'keep_prob': 0.724123179663981, 'accuracyDev': 0.8095238, 'accuracyTrain': 0.99946064}

    ## Units of layers
#     structure = [ 50, 24, 12, 1 ]
#     num_epochs = 1000
#     # Result from tuning
#     beta = 0
#     keep_prob = 1
#     learning_rate = 0.0001

    #structure = [ 100, 48, 1 ]
    # Result from tuning
    #beta = 1.6980624617370184e-15
    #keep_prob = 0.724123179663981

#     structure = [ 25, 12, 1 ]
#     # Result from tuning
#     beta = 6.531654400821318e-14
#     keep_prob = 0.8213956561201344
#     learning_rate = 0.0001
#     num_epochs = 1500

    # Loading the dataset
    X_train_orig, Y_train_orig, PATH_train, TAG_train, WEIGHT_train, \
    X_dev_orig  , Y_dev_orig, PATH_dev, TAG_dev= \
        load_dataset( hyperParams[ const.KEY_USE_WEIGHTS ] )

    # Flatten the training and test images
    X_train_flatten = X_train_orig.reshape( X_train_orig.shape[0], -1 ).T
    X_dev_flatten = X_dev_orig.reshape( X_dev_orig.shape[0], -1 ).T
    # Normalize image vectors
    X_train = X_train_flatten / 255.
    X_dev   = X_dev_flatten / 255.

    Y_train = Y_train_orig
    Y_dev = Y_dev_orig

    print ("number of training examples = " + str(X_train.shape[1]))
    print ("number of test examples = " + str(X_dev.shape[1]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_dev.shape))
    print ("Y_test shape: " + str(Y_dev.shape))
    print ()
    print ("Start Learning rate :", str( hyperParams[ const.KEY_START_LEARNING_RATE ] ) )
    print ("Num epoch           :", str( hyperParams[ const.KEY_NUM_EPOCHS ] ) )
    print ("Minibatch size      :", str( hyperParams[ const.KEY_MINIBATCH_SIZE ] ) )
    print ("Beta                :", str( hyperParams[ const.KEY_BETA ] ) )
    print ("keep_prob           :", str( hyperParams[ const.KEY_KEEP_PROB ] ) )
    print ("isLoadWeights :", hyperParams[ const.KEY_USE_WEIGHTS ] )
    if ( hyperParams[ const.KEY_USE_WEIGHTS ] ) :
        print ( "  Weights_train shape :", WEIGHT_train.shape )

    print( "Structure:", structure )

    dataInfo = {
        const.KEY_TRN_SIZE  : str( X_train.shape[1] ),
        const.KEY_DEV_SIZE       : str( X_dev.shape[1] ),
        const.KEY_TRN_SHAPE : str( X_train.shape ),
        const.KEY_DEV_SHAPE      : str( X_dev.shape ),
        const.KEY_TRN_Y_SIZE         : str( Y_dev.shape[1] ),
        const.KEY_TRN_Y_SHAPE        : str( Y_dev.shape ),
        const.KEY_DEV_Y_SIZE         : str( Y_dev.shape[1] ),
        const.KEY_DEV_Y_SHAPE        : str( Y_dev.shape ),
    }

    #
#    tuning( num_epochs = num_epochs, learning_rate = learning_rate )

    print()
    comment = input( "Run comment: " )

    # Init DB
    with db.initDb( APP_KEY, DB_DIR ) as conn:
    
        # Create run
        idRun = db.createRun( conn )
    
        # Update run before calling model
        db.updateRunBefore(
            conn, idRun,
            structure=structure, comment=comment,
            system_info=systemInfo, hyper_params=hyperParams, data_info=dataInfo
        )
    
        # Run model and update DB run with extra info
        model(
            conn, idRun, structure,
            X_train, Y_train, PATH_train, TAG_train, WEIGHT_train,
            X_dev, Y_dev, PATH_dev, TAG_dev,
            hyperParams
        )
    
        # Print run
        #run = db.getRun( conn, idRun )
        #print( "Run stored in DB:", str( run ) )

    print( "Finished" )
    