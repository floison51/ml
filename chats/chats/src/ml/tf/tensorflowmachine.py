'''
Created on 27 mai 2018

@author: fran
'''
import const.constants as const
from ml.machine import AbstractMachine

import tensorflow as tf
from tensorflow.python.framework import ops

import os as os

import math
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import time
import os
import shutil
from collections import OrderedDict

import db.db as db

# Abstract classes
import abc
from ml.tf.tfdatasource import TensorFlowDataSource

TENSORFLOW_SAVE_DIR = os.getcwd().replace( "\\", "/" ) + "/run/tf-save/"  + AbstractMachine.APP_KEY
TENSORBOARD_LOG_DIR = os.getcwd().replace( "\\", "/" ) + "/run/tf-board/" + AbstractMachine.APP_KEY

class AbstractTensorFlowMachine( AbstractMachine ):
    # Abstract class
    __metaclass__ = abc.ABCMeta

    def __init__( self, params = None ):
        AbstractMachine.__init__( self, params )
        self.isTensorboard     = False
        self.isTensorboardFull = False

    def setRunParams( self, runParams ):
        # tensorboard
        self.isTensorboard     = runParams[ "isTensorboard"     ]
        self.isTensorboardFull = runParams[ "isTensorboardFull" ]

    def addPerfInfo( self, perfInfo ):
        "Add perf information"

        super.addPerfInfo( self, perfInfo )

        perfInfo.append(
            { const.KEY_PERF_IS_USE_TENSORBOARD       : self.isTensorboard,
              const.KEY_PERF_IS_USE_FULL_TENSORBOARD  : self.isTensorboardFull,
            }
        )

    def getSession( self ):

        sess = tf.Session()
        return sess

    def getTensorBoardFolder( self, what ) :
        # root TFB folder + runID to get a name in TFB
        folder = TENSORBOARD_LOG_DIR + "/" + what + "/" + str( self.idRun )
        return folder

    def initSessionVariables( self, sess ):

        if ( self.isTensorboard ) :

            # delete files in run folder
            trnFolder = self.getTensorBoardFolder( "trn" )
            if tf.gfile.Exists( trnFolder ):
                tf.gfile.DeleteRecursively( trnFolder )
            tf.gfile.MakeDirs( trnFolder )

            self.trn_writer = tf.summary.FileWriter( trnFolder, sess.graph )

            devFolder = self.getTensorBoardFolder( "dev" )
            if tf.gfile.Exists( devFolder ):
                tf.gfile.DeleteRecursively( devFolder )
            tf.gfile.MakeDirs( devFolder )

            self.dev_writer = tf.summary.FileWriter( devFolder, sess.graph )

        # Run the initialization
        init = tf.global_variables_initializer()
        sess.run( init )


    @abc.abstractmethod
    def parseStructure( self, strStructure ):
        "Parse provided string structure into machine dependent model"

    def useDataSource( self ):
        return False

    def modelInit( self, strStructure, X_shape, X_type, Y_shape, Y_type, training ):

        tf.set_random_seed( 1 )                             # to keep consistent results

        # parse structure
        self.structure = self.parseStructure( strStructure )

        # Create Placeholders of shape (n_x, n_y)
        self.create_placeholders( X_shape, X_type, Y_shape, Y_type )

        # Initialize parameters
        self.initialize_parameters( X_shape )

        # Forward propagation: Build the forward propagation in the tensorflow graph
        ### START CODE HERE ### (1 line)
        Z_last = self.forward_propagation( X_shape, training )
        ### END CODE HERE ###

        # Cost function: Add cost function to tensorflow graph
        # TODO weight
        self.define_cost( Z_last, self.Y, 1 )

        # Back-propagation: Define the tensorflow optimizer. Use an AdamOptimizer.
        # Decaying learning rate
        global_step = tf.Variable( 0 )  # count the number of steps taken.

        learning_rate = tf.train.exponential_decay(
            self.start_learning_rate, global_step, self.learning_rate_decay_nb, self.learning_rate_decay_percent, staircase=True
        )

        # Adam optimizer
        self.optimizer = tf.train.AdamOptimizer( learning_rate ).minimize( self.cost, global_step = global_step )

        # Accuracy and correct prediction
        self.correct_prediction = self.defineAccuracy( self.Y, Z_last )

        # Tensorboard summaries
        with tf.name_scope( 'LearningRate' ):
            if ( self.isTensorboard ) :
                tf.summary.scalar( 'Num epoch', self.var_numEpoch )
                tf.summary.scalar( 'Step', global_step )
                tf.summary.scalar( 'learning_rate', learning_rate )

        with tf.name_scope( 'Result' ):
            if ( self.isTensorboard ) :
                tf.summary.scalar( 'Cost', self.cost )

                #Don'a dd accuracy as it is re-used DEV accuracy, this mismatches graph
                #tf.summary.scalar( 'TRN Accuracy', self.accuracy )
                tf.summary.scalar( 'DEV Accuracy', self.var_DEV_accuracy )

        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        self.mergedSummaries = tf.summary.merge_all()

        # Add ops to save and restore all the variables.
        self.tfSaver = tf.train.Saver()

    def modelEnd( self ):
        # Close tensorboard streams
        if ( self.trn_writer != None ) :
            self.trn_writer.close()
        if ( self.dev_writer != None ) :
            self.dev_writer.close()

    @abc.abstractmethod
    def define_cost( self, Z_last, Y, WEIGHT ):
        """
        Computes the cost

        Arguments:
        Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
        Y -- "true" labels vector placeholder, same shape as Z3

        Returns:
        cost - Tensor of the cost function
        """

    def defineAccuracy( self, Y, Z_last ):

        # To calculate the correct predictions
        prediction = tf.round( tf.sigmoid( Z_last ) )
        correct_prediction = tf.equal( prediction, Y )
        return correct_prediction

    def persistParams( self, sess, idRun ):

        # lets save the parameters in a variable
        #sess.run( self.parameters )

        # Serialize parameters
        save_dir = TENSORFLOW_SAVE_DIR + "/" + str( idRun ) + "/save"
        if tf.gfile.Exists( save_dir ):
            tf.gfile.DeleteRecursively( save_dir )
        tf.gfile.MakeDirs( save_dir )

        save_path = self.tfSaver.save( sess, save_dir )
        print( "Model saved in path: %s" % save_path)

    def getAccuracyEvalFeedDict( self, inputData ) :
        # Make sure KEEP_PROB = 1 and TRN_MODE = False
        feed_dict = { self.X: inputData[ 0 ], self.Y: inputData[ 1 ], self.ph_KEEP_PROB: 1.0, self.ph_TRN_MODE: False }
        return feed_dict

    def correctPredictionEval( self, inputData ):

        feed_dict = self.getAccuracyEvalFeedDict( inputData )

        sigmaCorrectPredictionInitialized = False
        sigmaCorrectPrediction = np.array( (), dtype=np.bool )

        if ( self.useDataSource() ) :

            # Iterate data source
            try:
                while ( True ) :
                    correct_predictions = self.correct_prediction.eval( feed_dict )
                    # Add terms
#                     if ( not sigmaCorrectPredictionInitialized ) :
#                         sigmaCorrectPrediction = correct_predictions
#                         sigmaCorrectPredictionInitialized = True
#                     else :
                    sigmaCorrectPrediction = np.append( sigmaCorrectPrediction, correct_predictions )

            except tf.errors.OutOfRangeError:
                # walk finished
                pass

        else :
            # One shot calculation
            sigmaCorrectPrediction = self.correct_prediction.eval( feed_dict )

        return sigmaCorrectPrediction

    def accuracyEval( self, inputData, what ) :

        correct_predictions = self.correctPredictionEval( inputData )

        # sum terms
        sigmaAccuracy = np.sum( correct_predictions )
        sigmaNb = correct_predictions.shape[ 0 ]

        # global accuracy
        accuracy = sigmaAccuracy / sigmaNb

        if ( what == "dev" ) :
            # Update accuracy dev variable
            self.var_DEV_accuracy.load( accuracy )

        return accuracy

    def create_placeholders( self, X_shape, X_type, Y_shape, Y_type ):
        "Creates the placeholders for the tensorflow session."

        self.ph_KEEP_PROB = tf.placeholder( tf.float32, name = "KEEP_PROB" )
        # Training mode
        self.ph_TRN_MODE  = tf.placeholder( tf.bool, name = "TRN_MODE" )

        self.var_numEpoch     = tf.get_variable( "NumEpoch"    , [] , dtype=tf.int32  , trainable=False )
        self.var_DEV_accuracy = tf.get_variable( "DEV_accuracy", [] , dtype=tf.float32, trainable=False )

    @abc.abstractmethod
    def initialize_parameters( self, X_shape ):
        "Initialize parameters given X lines dimension"

    @abc.abstractmethod
    def forward_propagation( self, X_shape, training ):
        "Define the forward propagation"

    def getRunIterationFeedDict( self, inputData, keep_prob ):
        feed_dict = { self.X: inputData[ 0 ], self.Y: inputData[ 1 ], self.ph_KEEP_PROB: keep_prob, self.ph_TRN_MODE: True }
        return feed_dict

    def runIteration(
        self, sess, input, keep_prob,
        iteration, num_minibatches
    ):

        feed_dict = self.getRunIterationFeedDict( input, keep_prob )

        # No mini-batch
        if ( self.isTensorboard and ( iteration % ( 100 * num_minibatches ) == 100 * num_minibatches - 1 ) ):  # Record execution stats
            run_options = tf.RunOptions( trace_level = tf.RunOptions.FULL_TRACE )
            run_metadata = tf.RunMetadata()
            summary, _ , curCost = sess.run(
                [ self.mergedSummaries, self.optimizer, self.cost ], feed_dict=feed_dict,
                options=run_options,run_metadata=run_metadata
            )
            self.trn_writer.add_run_metadata( run_metadata, 'step%03d' % iteration )
            self.trn_writer.add_summary( summary, iteration )
        else :

            if ( self.isTensorboard ) :
                #run without meta data
                summary, _ , curCost = sess.run(
                    [ self.mergedSummaries, self.optimizer, self.cost ], feed_dict=feed_dict
                )
                self.trn_writer.add_summary( summary, iteration )
            else :
               _ , curCost = sess.run(
                    [ self.optimizer, self.cost], feed_dict=feed_dict
                )

        return curCost

    def variable_summaries( self, var ):
      """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
      if ( self.isTensorboardFull ) :
          with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
              stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


#*****************************************************
# Tensorflow basic machine : supports [48,24,1] fully connected hand-mad structure
#*****************************************************
class TensorFlowSimpleMachine( AbstractTensorFlowMachine ):

    def __init__( self, params = None ):
        AbstractTensorFlowMachine.__init__( self, params )

    def parseStructure( self, strStructure ):
        ## Normalize structure
        strStructure = strStructure.strip()
        if strStructure[ 0 ] != "[" :
            raise ValueError( "Structure syntax: [48,24,1]" )

        if strStructure[ -1 ] != "]" :
            raise ValueError( "Structure syntax: [48,24,1]" )

        #Get as array
        structure = eval( strStructure )

        return structure

    def create_placeholders( self, X_shape, X_type, Y_shape, Y_type ):
        "Creates the placeholders for the tensorflow session."

        super().create_placeholders( X_shape, X_type, Y_shape, Y_type )

        self.X = tf.placeholder( X_type, shape=X_shape, name="X" )
        self.Y = tf.placeholder( Y_type, shape=Y_shape, name="Y" )

    def initialize_parameters( self, shape_X ):
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
        self.n_x = shape_X[ 1 ]
        structure0 = [ self.n_x ] + self.structure

        # parameters
        self.parameters = {}

        # browse layers
        for i in range( 0, len( structure0 ) - 1 ):

            # Adding a name scope ensures logical grouping of the layers in the graph.
            with tf.name_scope( "Layer" + str( i+1 ) ):
                # This Variable will hold the state of the weights for the layer
                with tf.name_scope( 'weights' ):
                    W_cur = tf.get_variable( "W" + str( i + 1 ), [ structure0[ i ], structure0[ i+1 ] ], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
                    if self.isTensorboardFull :
                        self.variable_summaries( W_cur )

                with tf.name_scope( 'bias' ):
                    b_cur = tf.get_variable( "b" + str( i + 1 ), [ 1, structure0[ i + 1 ] ], initializer = tf.zeros_initializer())
                    if self.isTensorboardFull :
                        self.variable_summaries( b_cur )

            self.parameters[ "W" + str( i + 1 ) ] = W_cur
            self.parameters[ "b" + str( i + 1 ) ] = b_cur

    def forward_propagation( self, X_shape, training ):
        """
        Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

        Arguments:
        X -- input dataset placeholder, of shape (input size, number of examples)
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                      the shapes are given in initialize_parameters

        Returns:
        Z3 -- the output of the last LINEAR unit
        """

        # check
        if ( self.useBatchNormalization ) :
            raise ValueError( "This machine doesn't support '" + const.KEY_USE_BATCH_NORMALIZATION + "' hyper-parameter" )

        ## Add Level 0 : X
        # example : 12228, 100, 24, 1
        n_x = X_shape[ 1 ]
        structure0 = [ n_x ] + self.structure

        Z = None
        A = None

        curInput = self.X

        for i in range( 1, len( structure0 ) ):
            # Adding a name scope ensures logical grouping of the layers in the graph.
            with tf.name_scope( "Layer" + str( i ) ):
                # apply DropOut to hidden layer
                curInput_drop_out = curInput
                if i < len( structure0 ) - 1 :
                    curInput_drop_out = tf.nn.dropout( curInput, self.ph_KEEP_PROB )

                ## Get W and b for current layer
                W_layer = self.parameters[ "W" + str( i ) ]
                b_layer = self.parameters[ "b" + str( i ) ]

                ## Linear part
                with tf.name_scope( 'Z' ):
                    Z = tf.add( tf.matmul( curInput_drop_out, W_layer ), b_layer )
                    if self.isTensorboardFull :
                        tf.summary.histogram( 'Z', Z )

                ## Activation function for hidden layers
                if i < len( structure0 ) - 1 :
                    A = tf.nn.relu( Z )
                    if self.isTensorboardFull :
                        tf.summary.histogram( 'A', A   )

                ## Change cur input to A(layer)
                curInput = A

        # For last layer (output layer): no dropout and return only Z
        return Z

    def define_cost( self, Z_last, Y, WEIGHT ):
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
        structure0 = [ self.n_x ] + self.structure

        # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)

        # if data samples per column
        #logits = tf.transpose( Z_last )
        #labels = tf.transpose( Y )

        # if data samples per line
        #ValueError: logits and labels must have the same shape ((1, 12288) vs (?, 1))
        logits = Z_last
        labels = Y

        raw_cost = None

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

        # Loss function using L2 Regularization
        regularizer = None

        if ( self.beta != 0 ) :
            losses = []
            for i in range( 1, len( structure0 ) ) :
                W_cur = self.parameters[ 'W' + str( i ) ]
                losses.append( tf.nn.l2_loss( W_cur ) )

            regularizer = tf.add_n( losses )

        cost = None

        if ( regularizer != None ) :
            cost = tf.reduce_mean( raw_cost + self.beta * regularizer )
        else :
            cost = raw_cost

        self.cost = cost

#*****************************************************
# Tensorflow machine : supports multi-layers tf notations, including convolution
#*****************************************************
class TensorFlowFullMachine( AbstractTensorFlowMachine ):

    def __init__( self, params = None ):
        super( AbstractTensorFlowMachine, self ).__init__( params )
        self.isTensorboard     = False
        self.isTensorboardFull = False

    def useDataSource( self ):
        return True

    def create_placeholders( self, X_shape, X_type, Y_shape, Y_type ):
        "Creates the placeholders for the tensorflow session."

        super().create_placeholders( X_shape, X_type, Y_shape, Y_type )

        #Num eopochs for trn dataset
        self.phTrnNumEpoch = tf.placeholder( tf.int64, name = "phTrnNumEpoch" )

        # Data set handle (human identifier)
        self.dsHandle = tf.placeholder(tf.string, shape=[], name="ph_Dataset" )

        # Iterator (X,Y)
        dsIterator = tf.data.Iterator.from_string_handle(
            self.dsHandle,
            output_types  = ( X_type , Y_type ),
            output_shapes = ( X_shape, Y_shape )
        )

        # X and Y vars
        ( self.X, self.Y ) = dsIterator.get_next()

    def parseStructure( self, strStructure ):
        ## Normalize structure
        strStructure = strStructure.strip()

        # Browse lines
        lines = strStructure.splitlines()
        isLastLine = False

        result = []

        for line in lines :

            if ( isLastLine ) :
                raise ValueError( "fullyConnected network must be last line" )
            line = line.strip()

            if ( not( line ) ) :
                # empty line
                continue;

            # FullyConnected network
            if ( line.startswith( "fullyConnected[" ) ) :

                # Last line
                isLastLine = True

                # parse fc network
                line = line[ len( "fullyConnected" ) : ]

                if line[ 0 ] != "[" :
                    raise ValueError( "Structure syntax: fullyConnected[48,24,1]" )

                if line[ -1 ] != "]" :
                    raise ValueError( "Structure syntax: fullyConnected[48,24,1]" )

                #Get as array
                structure = eval( line )
                result.append( ( "fullyConnected", structure ) )

            elif ( line.startswith( "conv2d(" ) or line.startswith( "max_pooling2d(" ) ) :
                # 2D convolution : conv2d( curInput, filters=32, kernel_size=[5, 5], padding="same",  activation=tf.nn.relu )
                # Add tf prefix
                line = "tf.layers." + line
                result.append( ( "tensor", line ) )

            elif ( line == "flatten" ) :
                result.append( ( "flatten", "" ) )

            else :
                raise ValueError( "Can't parse structure line '" + line )

        return result

    def initialize_parameters( self, X_shape ):
        "Not needed in this model"

    def forward_propagation( self, X_shape, training ):

        # Prepare normalizer tensor
        regularizer_l2 = None
        if ( self.beta != 0 ) :
            regularizer_l2 = tf.contrib.layers.l2_regularizer( self.beta )

        curInput = self.X

        # Browse structure
        for structureItem in self.structure :

            key   = structureItem[ 0 ]
            value = structureItem[ 1 ]

            if ( key == "tensor" ) :
                # value is a tensorwflow tensor
                AZ = eval( value )

            elif ( key == "flatten" ) :
                # flatten data
                # get input shape
                curShape = curInput.shape

                # get flat dimension
                flatDim = 1
                for dim in curShape.dims[ 1: ] :
                    flatDim *= dim.value

                AZ = tf.reshape( curInput, [-1, flatDim ] )

            elif ( key == "fullyConnected" ) :

                numLayers = value

                # browse layers
                iLayer = -1
                for numLayer in numLayers:
                    iLayer += 1
                    # last layer
                    lastLayer = ( iLayer == ( len( numLayers ) - 1 ) )

                    # filter input by keep prob if needed
                    if ( self.keep_prob != 1 and not lastLayer ) :
                        curInput = tf.contrib.layers.dropout( curInput, self.ph_KEEP_PROB )

                    # Z function
                    Z0 = tf.contrib.layers.fully_connected( \
                        inputs=curInput, num_outputs=numLayer, activation_fn=None, \
                        weights_initializer=tf.contrib.layers.xavier_initializer( seed = 1 ), \
                        weights_regularizer= regularizer_l2
                    )

                    if ( self.useBatchNormalization ) :
                        # Add batch normalization to speed-up gradiend descent convergence
                        # Not for training
                        Z1 = tf.layers.batch_normalization( Z0, training=self.ph_TRN_MODE )
                    else :
                        Z1 = Z0

                    if ( lastLayer ) :
                        # No activation function
                        AZ = Z1
                    else :
                        # activation function
                        AZ = tf.nn.relu( Z1 )

                    # next input in AZ
                    curInput = AZ

            else :
                raise ValueError( "Can't parse structure key '" + key + "'" )

            # next input in AZ
            curInput = AZ

        return AZ

    def define_cost( self, Z_last, Y, WEIGHT ):

        # if data samples per column
        #logits = tf.transpose( Z_last )
        #labels = tf.transpose( Y )

        # if data samples per line
        #ValueError: logits and labels must have the same shape ((1, 12288) vs (?, 1))
        logits = Z_last
        labels = Y

        raw_cost = None

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


        # Add regularization cost
        if ( self.beta != 0 ) :
            # Variables participating to regularization
            # regVariables = tf.get_collection( tf.GraphKeys.REGULARIZATION_LOSSES )
            # regCostTerm = tf.contrib.layers.apply_regularization()
            regCostTerm = tf.losses.get_regularization_loss()
            self.cost = raw_cost + regCostTerm

        else :
            self.cost = raw_cost

    def getRunIterationFeedDict( self, inputData, keep_prob ):
        feed_dict = { self.dsHandle: inputData, self.ph_KEEP_PROB: keep_prob, self.ph_TRN_MODE: True }
        return feed_dict

    def getAccuracyEvalFeedDict( self, inputData ) :
        # Make sure KEEP_PROB = 1 and TRN_MODE = False
        feed_dict = { self.dsHandle: inputData, self.ph_KEEP_PROB: 1.0, self.ph_TRN_MODE: False }
        return feed_dict


#*********************************************************************************************
# Estimation
#*********************************************************************************************
    def optimizeModel(
        self, conn, idRun,
        structure,
        hyperParams,
        print_cost = True, show_plot = True, extractImageErrors = True
    ):

        costs = []                                     # To keep track of the cost
        DEV_accuracies = []                            # for DEV accuracy graph

        # Get hyper parameters from dico
        self.beta           = hyperParams[ const.KEY_BETA ]
        self.keep_prob      = hyperParams[ const.KEY_KEEP_PROB ]
        self.num_epochs     = hyperParams[ const.KEY_NUM_EPOCHS ]
        self.minibatch_size = hyperParams[ const.KEY_MINIBATCH_SIZE ]

        self.start_learning_rate         = hyperParams[ const.KEY_START_LEARNING_RATE ]
        self.learning_rate_decay_nb      = hyperParams[ const.KEY_LEARNING_RATE_DECAY_NB ]
        self.learning_rate_decay_percent = hyperParams[ const.KEY_LEARNING_RATE_DECAY_PERCENT ]

        self.useBatchNormalization = hyperParams[ const.KEY_USE_BATCH_NORMALIZATION ]

        if ( self.minibatch_size < 0 ) :
            raise ValueError( "Mini-batch size is required" )

        ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables

        # Convert ( nbLines, dims... ) to ( None, dims... )
        X_shape = [ None ]
        X_shape.extend( self.dataInfo[ const.KEY_TRN_X_SHAPE ][ 1: ] )
        X_type = self.datasetTrn.X.dtype

        Y_shape = [ None ]
        Y_shape.extend( self.dataInfo[ const.KEY_TRN_Y_SHAPE ][ 1: ] )
        Y_type = self.datasetTrn.Y.dtype

        self.modelInit( structure, X_shape, X_type, Y_shape, Y_type, training=True )

        # Convert ( nbLines, dims... ) to ( None, dims... )
        self.tfDatasetTrn = tf.data.Dataset.from_tensor_slices(
            (
                self.datasetTrn.X,
                self.datasetTrn.Y,
            )
        )

        # Cache for performance
        self.tfDatasetTrn = self.tfDatasetTrn.cache()
        # Data set, repeat num_epochs, minibatch_size slices
        self.tfDatasetTrn = self.tfDatasetTrn.prefetch( self.minibatch_size * 2 ).batch( self.minibatch_size ).repeat( self.phTrnNumEpoch )

        self.tfDatasetDev = tf.data.Dataset.from_tensor_slices(
            (
                self.datasetDev.X,
                self.datasetDev.Y
            )
        )

        # Cache for performance
        self.tfDatasetDev = self.tfDatasetDev.cache()
        # Data set, repeat num_epochs, minibatch_size slices
        self.tfDatasetDev = self.tfDatasetDev.prefetch( self.minibatch_size * 2 ).batch( self.minibatch_size )

        trnIterator = self.tfDatasetTrn.make_initializable_iterator()
        devIterator = self.tfDatasetDev.make_initializable_iterator()

        # Start the session to compute the tensorflow graph
        with self.getSession() as sess:

            seed = 3 # to keep consistent results

            self.initSessionVariables( sess )

            # initialise variables iterators.
            sess.run( tf.global_variables_initializer() )
            sess.run( [ trnIterator.initializer, devIterator.initializer ], { self.phTrnNumEpoch : self.num_epochs } )

            # The `Iterator.string_handle()` method returns a tensor that can be evaluated
            # and used to feed the `handle` placeholder.
            trnHandle = sess.run( trnIterator.string_handle() )
            devHandle = sess.run( devIterator.string_handle() )

            ## optimisation may overshoot locally
            ## To avoid returning an overshoot, we detect it and run extra epochs if needed
            finalizationMode = False
            current_num_epochs = hyperParams[ const.KEY_NUM_EPOCHS ]
            minCost = 99999999999999
            minCostFinalization = 99999999999999
            finished = False

            # intercept Ctrl-C
            self.interrupted = False
            import signal
            # signal.signal( signal.SIGINT, self.signal_handler )

            # Minibatch mode, non handled by data source
            m = self.dataInfo[ const.KEY_TRN_X_SIZE ]              # m : number of examples in the train set)
            num_minibatches = math.ceil( m / self.minibatch_size ) # number of minibatches of size minibatch_size in the train set

            # Do the training loop
            iEpoch = 0
            minibatch_cost = 0
            epoch_cost = 0.                       # Defines a cost related to an epoch
            # current iteration
            iteration = 0

            # Start time
            tsStart = time.time()

            # time to make sure we trace something each N minuts
            tsTraceStart = tsStart

            try :
                while ( not self.interrupted and not finished ) :

                    minibatch_cost = self.runIteration(
                        sess, trnHandle, self.keep_prob, iteration, num_minibatches
                    )

                    epoch_cost += minibatch_cost / num_minibatches

                    if ( print_cost and iteration == 0 ) :
                        # Display iteration 0 to allow verify cost calculation accross machines
                        print ( "TRACE : Current cost epoch %i; iteration %i; %f" % ( iEpoch, iteration, epoch_cost ) )

                    # time to trace?
                    tsTraceNow = time.time()
                    tsTraceElapsed = tsTraceNow - tsTraceStart

                    # Each 60 seconds
                    if ( tsTraceElapsed >= 60 ) :

                        # Display iteration 0 to allow verify cost calculation accross machines
                        print ( "TRACE : Current cost epoch %i; iteration %i; %f" % ( iEpoch, iteration, epoch_cost ) )
                        # reset trace start
                        tsTraceStart = tsTraceNow

                    # Current epoch finished?
                    if ( ( iteration + 1 ) % num_minibatches == 0 ) :

                        # Load epoch in tensorboard
                        self.var_numEpoch.load( iEpoch )

                        #print epoch cost
                        if print_cost and ( iteration != 0 ) and ( iEpoch % 1 ) == 0:
                            print ("Cost after epoch %i; iteration %i; %f" % ( iEpoch, iteration, epoch_cost ) )
                            if ( iEpoch != 0 ) :

                                # Performance counters
                                curElapsedSeconds, curPerfIndex = self.getPerfCounters( tsStart, iEpoch, self.datasetTrn.X.shape )
                                print( "  current: elapsedTime; %i; perfIndex; %f" % ( curElapsedSeconds, curPerfIndex ) )

                                #  calculate DEV accuracy
                                # Rewind DEV iterator
                                sess.run( [ devIterator.initializer ] )
                                DEV_accuracy = self.accuracyEval( devHandle, "dev" )
                                print( "  current: DEV accuracy: %f" % ( DEV_accuracy ) )
                                DEV_accuracies.append( DEV_accuracy )

                        # Store cost for graph
                        if print_cost == True and ( iteration != 0 ) and iEpoch % 5 == 0:
                            costs.append( epoch_cost )

                        # Record min cost
                        minCost = min( minCost, epoch_cost )

                        # epoch changed
                        iEpoch += 1
                        epoch_cost = 0

                    # Close to finish?
#                     if ( not finalizationMode and ( iEpoch > current_num_epochs ) ) :
#                         # Activate finalization mode
#                         finalizationMode = True
#                         # local overshoot?
#                         if ( epoch_cost > minCost ) :
#                             # Yes, run some extra epochs
#                             print( "WARNING: local cost overshoot detected, adding maximum 100 epochs to leave local cost overshoot" )
#                             current_num_epochs += 100
#                             minCostFinalization = minCost
#
#                     if ( finalizationMode ) :
#                         # Check overshoot is finished
#                         if ( epoch_cost <= minCostFinalization ) :
#                             # finished
#                             finished = True

                    iteration += 1

            except tf.errors.OutOfRangeError:
                # walk finished
                pass

            self.modelOptimizeEnd( sess )

            if ( self.interrupted ) :
                print( "Training has been interrupted by Ctrl-C" )
                print( "Store current epoch number '" + str( iEpoch ) + "' in run hyper parameters" )
                # Get runs and hps
                run = db.getRun( conn, self.idRun )
                idRunHps = run[ "idHyperParams" ]
                runHps = db.getHyperParams( conn, idRunHps )[ "hyperParameters" ]
                # Modify num epochs
                runHps[ const.KEY_NUM_EPOCHS ] = iEpoch
                # update run
                db.updateRun( conn, self.idRun, runHps )

            # Final cost
            print ("Parameters have been trained!")
            print( "Final cost:", epoch_cost )

            ## Elapsed (seconds)
            elapsedSeconds, perfIndex = self.getPerfCounters( tsStart, iEpoch, self.datasetTrn.X.shape )
            perfInfo = {}

            print( "Elapsed (s):", elapsedSeconds )
            print( "Perf index :", perfIndex )

            self.persistParams( sess, idRun )

            # Rewind data sets, 1 epoch for TRN data set
            sess.run( [ trnIterator.initializer, devIterator.initializer ], { self.phTrnNumEpoch : 1 } )

            accuracyTrain = self.accuracyEval( trnHandle, "trn" )
            print ( "Train Accuracy:", accuracyTrain )

            accuracyDev = self.accuracyEval( devHandle, "dev" )
            print ( "Dev Accuracy:", accuracyDev )

            if ( show_plot ) :
                # plot the cost
                plt.plot(np.squeeze(costs))
                plt.ylabel('cost')
                plt.xlabel('iterations (per tens)')
                plt.title("Start learning rate =" + str( self.start_learning_rate ) )
                plt.show()

                # plot the accuracies
                plt.plot( np.squeeze( DEV_accuracies ) )
                plt.ylabel('DEV accuracy')
                plt.xlabel('iterations (100)')
                plt.title("Start learning rate =" + str( self.start_learning_rate ) )
                plt.show()

            ## Errors
            resultInfo = {}

            if ( extractImageErrors ) :

                # Rewind data sets, 1 epoch for TRN data set
                sess.run( [ trnIterator.initializer, devIterator.initializer ], { self.phTrnNumEpoch : 1 } )

                # Lists of OK for training
                oks_train  = self.correctPredictionEval( trnHandle )
                map1, map2 = self.statsExtractErrors( "train", dataset = self.datasetTrn, oks = oks_train, show_plot=show_plot )
                # Errors nb by data tag
                resultInfo[ const.KEY_TRN_NB_ERROR_BY_TAG ] = map1
                resultInfo[ const.KEY_TRN_PC_ERROR_BY_TAG ] = map2

                oks_dev   = self.correctPredictionEval( trnHandle )
                map1, map2 = self.statsExtractErrors( "dev", dataset = self.datasetDev, oks = oks_dev, show_plot=show_plot )
                # Errors nb by data tag
                resultInfo[ const.KEY_DEV_NB_ERROR_BY_TAG ] = map1
                resultInfo[ const.KEY_DEV_PC_ERROR_BY_TAG ] = map2

            # Update DB run after execution, add extra info
            db.updateRunAfter(
                conn, idRun,
                perf_info = perfInfo, result_info=resultInfo,
                perf_index=perfIndex,
                elapsed_second = elapsedSeconds,
                train_accuracy=accuracyTrain.astype( float ),
                dev_accuracy=accuracyDev.astype( float )
            )

            return accuracyDev, accuracyTrain

