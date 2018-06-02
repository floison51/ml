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

    def getTensorBoardFolder( self, what ) :
        # root TFB folder + runID to get a name in TFB
        folder = TENSORBOARD_LOG_DIR + "/" + what + "/" + str( self.idRun )
        return folder

    def sessionInit( self, sess ):

        if ( self.isTensorboard ) :
            self.trn_writer = tf.summary.FileWriter( self.getTensorBoardFolder( "trn" ), sess.graph )
            self.dev_writer = tf.summary.FileWriter( self.getTensorBoardFolder( "dev" ), sess.graph )

        # Run the initialization
        init = tf.global_variables_initializer()
        sess.run( init )

    @abc.abstractmethod
    def parseStructure( self, strStructure ):
        "Parse provided string structure into machine dependent model"

    def modelInit( self, strStructure, X_shape, Y_shape, training ):

        ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
        tf.set_random_seed( 1 )                             # to keep consistent results

        # parse structure
        self.structure = self.parseStructure( strStructure )

        # Create Placeholders of shape (n_x, n_y)
        self.create_placeholders( X_shape, Y_shape )

        # Initialize parameters
        self.initialize_parameters( X_shape )

        # Forward propagation: Build the forward propagation in the tensorflow graph
        ### START CODE HERE ### (1 line)
        Z_last = self.forward_propagation( X_shape, training )
        ### END CODE HERE ###

        # Cost function: Add cost function to tensorflow graph
        self.define_cost( Z_last, self.ph_Y, self.datasetTrn.weight )

        # Back-propagation: Define the tensorflow optimizer. Use an AdamOptimizer.
        # Decaying learning rate
        global_step = tf.Variable( 0 )  # count the number of steps taken.

        learning_rate = tf.train.exponential_decay(
            self.start_learning_rate, global_step, self.learning_rate_decay_nb, self.learning_rate_decay_percent, staircase=True
        )

        # Adam optimizer
        self.optimizer = tf.train.AdamOptimizer( learning_rate ).minimize( self.cost, global_step = global_step )

        # Accuracy and correct prediction
        self.accuracy, self.correct_prediction = self.defineAccuracy( self.ph_Y, Z_last )

        # Tensorboard summaries
        with tf.name_scope( 'LearningRate' ):
            if ( self.isTensorboard ) :
                tf.summary.scalar( 'Step', global_step )
                tf.summary.scalar( 'learning_rate', learning_rate )

        with tf.name_scope( 'Result' ):
            if ( self.isTensorboard ) :
                tf.summary.scalar( 'Cost', self.cost )
                tf.summary.scalar( 'TRN Accuracy', self.accuracy )
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
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        return accuracy, correct_prediction

    def devAccuracyUpdated( self, devAccuracy ):
        "Updated accuracy"
        # save it placeholder to have graph in tensorboard
        tf.assign( self.var_DEV_accuracy, devAccuracy )

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

    def create_placeholders( self, X_shape, Y_shape ):
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

        self.ph_X         = tf.placeholder( tf.float32, shape=X_shape, name = "X" )
        self.ph_Y         = tf.placeholder( tf.float32, shape=Y_shape, name = "Y" )
        self.ph_KEEP_PROB = tf.placeholder( tf.float32, name = "KEEP_PROB" )

        # DEV accuracy estimation
        self.var_DEV_accuracy = tf.get_variable( "DEV_accuracy", [] , dtype=tf.float32, trainable=False )

        # Assign value
        self.var_DEV_accuracy.assign( 0 )

    @abc.abstractmethod
    def initialize_parameters( self, X_shape ):
        "Initialize parameters given X lines dimension"

    @abc.abstractmethod
    def forward_propagation( self, X_shape, training ):
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
            self.trn_writer.add_run_metadata( run_metadata, 'step%03d' % iteration )
            self.trn_writer.add_summary( summary, iteration )
        else :

            if ( self.isTensorboard ) :
                #run without meta data
                summary, _ , curCost = sess.run(
                    [ self.mergedSummaries, self.optimizer, self. cost ], feed_dict=feed_dict
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

        curInput = self.ph_X

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
                        Z1 = tf.layers.batch_normalization( Z0, training=training )
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
