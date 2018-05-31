'''
Created on 27 mai 2018

@author: fran
'''
import const.constants as const
from ml.machine import AbstractMachine

import tensorflow as tf
from tensorflow.python.framework import ops

import os as os

# Abstract classes
import abc

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
        self.sessionInit( sess )
        return sess

    def sessionInit( self, sess ):
        self.train_writer = tf.summary.FileWriter( TENSORBOARD_LOG_DIR + '/train', sess.graph )
        self.dev_writer   = tf.summary.FileWriter( TENSORBOARD_LOG_DIR + '/dev', sess.graph )

        # Run the initialization
        init = tf.global_variables_initializer()
        sess.run( init )

    @abc.abstractmethod
    def parseStructure( self, strStructure ):
        "Parse provided string structure into machine dependent model"

    def modelInit( self, strStructure, n_x, n_y ):

        ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
        tf.set_random_seed( 1 )                             # to keep consistent results

        # parse structure
        self.structure = self.parseStructure( strStructure )

        # Create Placeholders of shape (n_x, n_y)
        self.create_placeholders( n_x, n_y )

        # Initialize parameters
        self.initialize_parameters( n_x )

        # Forward propagation: Build the forward propagation in the tensorflow graph
        ### START CODE HERE ### (1 line)
        Z_last = self.forward_propagation( n_x )
        ### END CODE HERE ###

        # Cost function: Add cost function to tensorflow graph
        self.define_cost( Z_last, self.ph_Y, self.datasetTrn.weight, self.beta, n_x )

        # Back-propagation: Define the tensorflow optimizer. Use an AdamOptimizer.
        # Decaying learning rate
        global_step = tf.Variable( 0 )  # count the number of steps taken.

        with tf.name_scope('cross_entropy'):
            with tf.name_scope( 'total' ):
                learning_rate = tf.train.exponential_decay(
                    # todo : use hp
                    self.start_learning_rate, global_step, 10000, 0.96, staircase=True
                )
                tf.summary.scalar( 'learning_rate', learning_rate )

        # fixed learning rate
        # learning_rate = start_learning_rate

        # Adam optimizer
        self.optimizer = tf.train.AdamOptimizer( learning_rate ).minimize( self.cost, global_step = global_step )

        # Accuracy and correct prediction
        self.accuracy, self.correct_prediction = self.defineAccuracy( self.ph_Y, Z_last )

        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        self.mergedSummaries = tf.summary.merge_all()

        # Add ops to save and restore all the variables.
        self.tfSaver = tf.train.Saver()

    def modelEnd( self ):
        # Close tensorboard streams
        self.train_writer.close()
        self.dev_writer.close()

    @abc.abstractmethod
    def define_cost( self, Z_last, Y, WEIGHT, beta, n_x ):
        """
        Computes the cost

        Arguments:
        Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
        Y -- "true" labels vector placeholder, same shape as Z3

        Returns:
        cost - Tensor of the cost function
        """

    def defineAccuracy( self, Y, Z_last ):
        with tf.name_scope('accuracy'):
            # To calculate the correct predictions
            prediction = tf.round( tf.sigmoid( Z_last ) )
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal( prediction, Y )
            with tf.name_scope('accuracy'):
                # Calculate accuracy on the test set
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        if self.isTensorboard :
            tf.summary.scalar('accuracy', accuracy)

        return accuracy, correct_prediction

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

    def accuracyEval( self, X, Y ):
        accuracy = self.accuracy.eval( { self.ph_X: X, self.ph_Y: Y, self.ph_KEEP_PROB: 1.0 } )
        return accuracy

    def correctPredictionEval( self, X, Y ):
        correct_prediction = self.correct_prediction.eval( { self.ph_X: X, self.ph_Y: Y, self.ph_KEEP_PROB: 1.0 } )
        return correct_prediction

    def create_placeholders( self, n_x, n_y ):
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
        self.ph_X         = tf.placeholder( tf.float32, shape=( None, n_x ), name = "X" )
        self.ph_Y         = tf.placeholder( tf.float32, shape=( None, n_y ), name = "Y" )
        self.ph_KEEP_PROB = tf.placeholder( tf.float32, name = "KEEP_PROB" )
        ### END CODE HERE ###

    @abc.abstractmethod
    def initialize_parameters( self, n_x ):
        "Initialize parameters given X lines dimension"

    @abc.abstractmethod
    def forward_propagation( self, n_x ):
        "Define the forward propagation"

    def runIteration(
        self,
        iteration, num_minibatches, sess,
        X, Y, keep_prob,
    ):

        feed_dict = { self.ph_X: X, self.ph_Y: Y, self.ph_KEEP_PROB: keep_prob }

        # No mini-batch
        if ( self.isTensorboard and ( iteration % ( 100 * num_minibatches ) == 100 * num_minibatches - 1 ) ):  # Record execution stats
            run_options = tf.RunOptions( trace_level = tf.RunOptions.FULL_TRACE )
            run_metadata = tf.RunMetadata()
            summary, _ , curCost = sess.run(
                [ self.mergedSummaries, self.optimizer, self.cost ], feed_dict=feed_dict,
                options=run_options,run_metadata=run_metadata
            )
            self.train_writer.add_run_metadata( run_metadata, 'step%03d' % iteration )
            self.train_writer.add_summary( summary, iteration )
        else :

            if ( self.isTensorboard ) :
                #run without meta data
                summary, _ , curCost = sess.run(
                    [ self.mergedSummaries, self.optimizer, self. cost ], feed_dict=feed_dict
                )
                self.train_writer.add_summary( summary, iteration )
            else :
                _ , curCost = sess.run(
                    [ self.optimizer, self.cost], feed_dict=feed_dict
                )

        return curCost

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

    def initialize_parameters( self, n_x ):
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
        structure0 = [ n_x ] + self.structure

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

    def forward_propagation( self, n_x ):
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
        structure0 = [ n_x ] + self.structure

        Z = None
        A = None

        curInput = self.ph_X

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

    def define_cost( self, Z_last, Y, WEIGHT, beta, n_x ):
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
        structure0 = [ n_x ] + self.structure

        with tf.name_scope('cross_entropy'):

            # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)

            # if data samples per column
            #logits = tf.transpose( Z_last )
            #labels = tf.transpose( Y )

            # if data samples per line
            #ValueError: logits and labels must have the same shape ((1, 12288) vs (?, 1))
            logits = Z_last
            labels = Y

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
                        W_cur = self.parameters[ 'W' + str( i ) ]
                        losses.append( tf.nn.l2_loss( W_cur ) )

                    regularizer = tf.add_n( losses )

                cost = None

                if ( regularizer != None ) :
                    cost = tf.reduce_mean( raw_cost + beta * regularizer )
                else :
                    cost = raw_cost

                tf.summary.scalar( 'cost', cost)

        self.cost = cost

#*****************************************************
# Tensorflow machine : supports multi-layers tf notations, including convolution
#*****************************************************
class TensorFlowFullMachine( AbstractTensorFlowMachine ):

    def __init__( self, params = None ):
        super( AbstractTensorFlowMachine, self ).__init__( params )
        self.isTensorboard     = False
        self.isTensorboardFull = False

    def parseStructure( self, strStructure ):
        ## Normalize structure
        strStructure = strStructure.strip()

        # Browse lines
        lines = strStructure.split( "\r" )
        isLastLine = False

        result = []

        for line in lines :

            if ( isLastLine ) :
                raise ValueError( "fullyConnected network must be last line" )
            line = line.strip()

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

        return result

    def initialize_parameters( self, n_x ):
        "Not needed in this model"

    def forward_propagation( self, n_x ):

        # Prepare normalizer tensor
        regularizer_l2 = None
        if ( self.beta != 0 ) :
            regularizer_l2 = tf.contrib.layers.l2_regularizer( self.beta )

        curInput = self.ph_X

        # Browse structure
        for structureItem in self.structure :

            if ( structureItem[ 0 ] == "fullyConnected" ) :
                numLayers = structureItem[ 1 ]
                # add input layer
                numLayers.insert( 0, n_x )
                
                # browse layers
                iLayer = -1
                for numLayer in numLayers:
                    iLayer += 1
                    # last layer
                    lastLayer = ( iLayer == ( len( numLayers ) - 1 ) )
                    
                    # filter input by keep prob if needed
                    if ( self.keep_prob != 1 ) :
                        curInput = tf.contrib.layers.dropout( curInput, self.ph_KEEP_PROB )
                    
                    # Activation function
                    if ( lastLayer ) :
                        activationFunction = None
                    else :
                        activationFunction = tf.nn.relu
                        
                    # current fully connected layer
                    Z = tf.contrib.layers.fully_connected( \
                        inputs=curInput, num_outputs=numLayer, activation_fn=activationFunction, \
                        weights_initializer=tf.contrib.layers.xavier_initializer( seed = 1 ), \
                        weights_regularizer= regularizer_l2
                    )
                    
                    # next input in Z
                    curInput = Z

        return Z

    def define_cost( self, Z_last, Y, WEIGHT, beta, n_x ):

        with tf.name_scope('cross_entropy'):

            # if data samples per column
            #logits = tf.transpose( Z_last )
            #labels = tf.transpose( Y )

            # if data samples per line
            #ValueError: logits and labels must have the same shape ((1, 12288) vs (?, 1))
            logits = Z_last
            labels = Y

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

        # Add regularization cost
        if ( self.beta != 0 ) :
            # Variables participating to regularization
            # regVariables = tf.get_collection( tf.GraphKeys.REGULARIZATION_LOSSES )
            # regCostTerm = tf.contrib.layers.apply_regularization()
            regCostTerm = tf.losses.get_regularization_loss()
            self.cost = raw_cost + regCostTerm

        else :
            self.cost = raw_cost
