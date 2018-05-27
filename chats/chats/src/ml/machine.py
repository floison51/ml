'''
Created on 25 mai 2018

@author: frup82455
'''
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import time
import os
import shutil
from collections import OrderedDict

import const.constants as const
import db.db as db
from setuptools.msvc import SystemInfo

class Machine():

    APP_KEY = "chats"

    '''
    Abstract Machine Learning class
    '''

    def __init__( self, params = None ):
        '''
        Constructor
        '''
        

    def addSystemInfo( self, systemInfo ):
        "Add system information"
        # TODO
    
    def addPerfInfo( self, systemInfo ):
        "Add perf information"
    
    def setInfos( self, systemInfo, dataInfo ):
        self.systemInfo = systemInfo
        self.dataInfo = dataInfo
        
    def setData(self, datasetTrn, datasetDev ):
        self.datasetTrn  = datasetTrn
        self.datasetDev  = datasetDev
        
    def train( self,  conn, config, comment, tune = False ):
        "Train the model"

        # hyper parameters
        confHyperParams = config.getHyperParams( conn )
        
        runHyperParams = {}
        runHyperParams.update( confHyperParams[ "hyperParameters" ] )

        ## Prepare hyper params
        if tune : 
            # Tune params
            beta_min = 0.000000000000001
            beta_max = 0.5
            
            keep_prob_min = 0.5
            keep_prob_max = 1
        
            nbTuning = 20
            tuning= {}
            
            maxAccuracyDev = -9999999999999
            maxIdRun = -1
        else :
            nbTuning = 1
        
        # Display hyper parameters info
        print ("Start Learning rate :", str( runHyperParams[ const.KEY_START_LEARNING_RATE ] ) )
        print ("Num epoch           :", str( runHyperParams[ const.KEY_NUM_EPOCHS ] ) )
        print ("Minibatch size      :", str( runHyperParams[ const.KEY_MINIBATCH_SIZE ] ) )
        print ("Beta                :", str( runHyperParams[ const.KEY_BETA ] ) )
        print ("keep_prob           :", str( runHyperParams[ const.KEY_KEEP_PROB ] ) )
        print ("isLoadWeights       :", runHyperParams[ const.KEY_USE_WEIGHTS ] )

        # Start time
        tsGlobalStart = time.time()
       
        for j in range( 1, nbTuning + 1 ) :
            
            if tune:
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
                
                # update hyper params
                runHyperParams[ const.KEY_BETA         ] = beta
                runHyperParams[ const.KEY_KEEP_PROB    ] = keep_prob
        
            # Create run
            
            idRun = db.createRun( conn, config[ "id" ],  runHyperParams )
    
            # Update run before calling model
            db.updateRunBefore(
                conn, idRun,
                comment=comment,
                system_info=self.systemInfo, data_info=self.dataInfo
            )
    
            # Run model and update DB run with extra info
            accuracyDev, accuracyTrain = self.model(
                conn, idRun,
                config[ "structure" ],
                runHyperParams,
                show_plot = not tune, extractImageErrors = not tune
            )
    
            # Print run
            run = db.getRun( conn, idRun )
            print( "Run stored in DB:", str( run ) )

            if tune :
                # Store results
                tuning[ j ] = { 
                    "beta": beta, "keep_prob": keep_prob, 
                    "accuracyDev": accuracyDev, "accuracyTrain": accuracyTrain
                }
            
                # Max
                if ( accuracyDev > maxAccuracyDev ) :
                    maxAccuracyDev = accuracyDev
                    maxHyperParams = tuning[ j ]
                    maxIdRun = idRun
                    
                # print max
                print( "Max DEV accuracy:", maxAccuracyDev )
                print( "Max hyper params:" )
                print( maxHyperParams )
            
                
        if tune :
            # Print tuning
            print( "Tuning:" , tuning )
            print()
            print( "Max DEV accuracy      :", maxAccuracyDev )
            print( "Max hyper params idRun:", maxIdRun )
    
        # Start time
        tsGlobalEnd = time.time()   
        globalElapsedSeconds = int( round( tsGlobalEnd - tsGlobalStart ) )
    
        print( "Finished in", globalElapsedSeconds, "seconds" )

    def modelInit( self, n_x, n_y ):
        # Abstract methode
        raise ValueError( "Abstract method" )
    
    def getSession( self ):
        # Abstract methode
        raise ValueError( "Abstract method" )
          
    def sessionInit( self, sess ):
        # Abstract methode
        raise ValueError( "Abstract method" )
          
    def modelEnd( self ):
        # Abstract methode
        raise ValueError( "Abstract method" )
    
    def persistParams( self ):
        # Abstract methode
        raise ValueError( "Abstract method" )
    
    def accuracyEval( self, X, Y ):
        # Abstract methode
        raise ValueError( "Abstract method" )
    
    def correctPredictionEval( self, X, Y ):
        # Abstract methode
        raise ValueError( "Abstract method" )
    
    def model(
        self, conn, idRun,
        structure,
        hyperParams,
        print_cost = True, show_plot = True, extractImageErrors = True
    ):
        
        (n_x, m) = self.datasetTrn.X.shape   # (n_x: input size, m : number of examples in the train set)
        n_y = self.datasetTrn.Y.shape[ 0 ]   # n_y : output size
        costs = []                                        # To keep track of the cost
    
        # Get hyper parameters from dico
        self.beta        = hyperParams[ const.KEY_BETA ]
        self.keep_prob   = hyperParams[ const.KEY_KEEP_PROB ]
        self.num_epochs  = hyperParams[ const.KEY_NUM_EPOCHS ]
        self.minibatch_size      = hyperParams[ const.KEY_MINIBATCH_SIZE ]
        self.start_learning_rate = hyperParams[ const.KEY_START_LEARNING_RATE ]
    
        self.modelInit( structure, n_x, n_y )

        seed = 3 # to keep consistent results

        # Start time
        tsStart = time.time()
    
        # Start the session to compute the tensorflow graph
        
        with self.getSession() as sess:
    
            # init session
            self.sessionInit( sess )
            
            # current iteration
            iteration = 0
    
            ## optimisation may overshoot locally
            ## To avoid returning an overshoot, we detect it and run extra epochs if needed
            finalizationMode = False
            current_num_epochs = hyperParams[ const.KEY_NUM_EPOCHS ]
            iEpoch = 0
            minCost = 99999999999999
            minCostFinalization = 99999999999999
            finished = False
    
            # Do the training loop
            while ( not finished and ( iEpoch <= current_num_epochs ) ) :
    
                epoch_cost = 0.                       # Defines a cost related to an epoch
                
                if ( self.minibatch_size < 0 ) :
    
                    # No mini-batch : do a gradient descent for whole data
                    
                    epoch_cost = self.runIteration(
                        iEpoch, 1, sess,
                        self.datasetTrn.X, self.datasetTrn.Y, self.keep_prob,
                    )
    
                    iteration += 1
    
                else:
                    #Minibatch mode
                    num_minibatches = int( m / self.minibatch_size ) # number of minibatches of size minibatch_size in the train set
                    seed = seed + 1
                    minibatches = self.random_mini_batches( self.datasetTrn.X, self.datasetTrn.Y, self.minibatch_size, seed )
    
                    for minibatch in minibatches:
    
                        # Select a minibatch
                        (minibatch_X, minibatch_Y) = minibatch
    
                        minibatch_cost = self.runIteration(
                            iteration, num_minibatches, sess,
                            minibatch_X, minibatch_Y, self.keep_prob
                        )
    
                        epoch_cost += minibatch_cost / num_minibatches
                        iteration += 1
    
                if print_cost == True and iEpoch % 100 == 0:
                    print ("Cost after epoch %i, iteration %i: %f" % ( iEpoch, iteration, epoch_cost ) )
                    if ( iEpoch != 0 ) :
                        # Performance counters
                        curElapsedSeconds, curPerfIndex = self.getPerfCounters( tsStart, iEpoch, n_x, m )
                        print( "  current: elapsedTime:", curElapsedSeconds, "perfIndex:", curPerfIndex ) 
                    
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
    
            self.modelEnd()
    
            # Final cost
            print ("Parameters have been trained!")
            print( "Final cost:", epoch_cost )
    
            # End time
            tsEnd = time.time()
    
            if ( show_plot ) :
                # plot the cost
                plt.plot(np.squeeze(costs))
                plt.ylabel('cost')
                plt.xlabel('iterations (per tens)')
                plt.title("Start learning rate =" + str( self.start_learning_rate ) )
                plt.show()
    
            self.persistParams( sess, idRun )
    
            accuracyTrain = self.accuracyEval( self.datasetTrn.X, self.datasetTrn.Y )
            print ( "Train Accuracy:", accuracyTrain )
    
            accuracyDev = self.accuracyEval( self.datasetDev.X, self.datasetDev.Y )
            print ( "Dev Accuracy:", accuracyDev )
    
            ## Elapsed (seconds)
            elapsedSeconds, perfIndex = self.getPerfCounters( tsStart, iEpoch, n_x, m, tsEnd )
            perfInfo = {}
        
            print( "Elapsed (s):", elapsedSeconds )
            print( "Perf index :", perfIndex )
        
            ## Errors
            resultInfo = {}
    
            if ( extractImageErrors ) :
    
                # Lists of OK for training
                oks_train  = self.correctPredictionEval(  self.datasetTrn.X, self.datasetTrn.Y )
                map1, map2 = self.statsExtractErrors( "train", dataset = self.datasetTrn, oks = oks_train, show_plot=show_plot )
                # Errors nb by data tag
                resultInfo[ const.KEY_TRN_NB_ERROR_BY_TAG ] = map1
                resultInfo[ const.KEY_TRN_PC_ERROR_BY_TAG ] = map2
     
                oks_dev   = self.correctPredictionEval(  self.datasetDev.X, self.datasetDev.Y )
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

    def runIteration(
        self,
        iteration, num_minibatches, sess,
        X, Y, keep_prob
    ):
        # Abstract methode
        raise ValueError( "Abstract method" )
        
    def random_mini_batches( self, X, Y, mini_batch_size = 64, seed = 0):
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

    def getPerfCounters( self, tsStart, iEpoch, n_x, m, tsNow = time.time() ):
        
        ## Elapsed (seconds)
        elapsedSeconds = int( round( tsNow - tsStart ) )
        # performance index : per iEpoth - per samples
        perfIndex = 1 / ( elapsedSeconds / iEpoch / ( n_x * m ) ) * 1e-6
        
        return elapsedSeconds, perfIndex
    
    def dumpBadImages( self, correct, X_orig, PATH, TAG, errorsDir ):
    
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
    
        # Dico of errors by label
        mapErrorNbByTag = {}
    
        imgBase = os.getcwd().replace( "\\", "/" ) + "/data/transformed"
    
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
    
                # extract 64x64x3 image
                #X_errorImg = X_orig[ i ]
                #errorImg = Image.fromarray( X_errorImg, 'RGB' )
    
                ## dump image
                #errorImg.save( errorsDir + '/error-' + str( i ) + ".png", 'png' )
    
                # Get original image
                # str: b'truc'
                imgRelPath = str( PATH[ i ] )
                # b'truc'
                imgRelPath = imgRelPath[ 2: ]
                # truc
                imgRelPath = imgRelPath[ : -1 ]
    
                imgPath = imgBase + "/" + imgRelPath
    
                toFile = errorsDir + "/" + label + "-" + str( i ) + "-" + os.path.basename( imgRelPath )
                shutil.copyfile( imgPath, toFile )
    
        # return dico
        return mapErrorNbByTag

    def statsExtractErrors( self, key, dataset, oks, show_plot=True ) :
    
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
        mapErrorNbByTag = self.dumpBadImages( oks, dataset.X_ori, dataset.imgPath, dataset.tag, errorsDir )
    
        # Sort by value
        mapErrorNbByTagSorted = \
            OrderedDict(
                sorted( mapErrorNbByTag.items(), key=lambda t: t[1], reverse=True )
        )
    
        ## Error repartition by label
        print( "Nb errors by tag for", key, ": ", mapErrorNbByTagSorted )
    
        # Build %age map
        nbSamples = oks.shape[ 1 ]
    
        mapErrorPercentNbByTag = {}
    
        for labelError in mapErrorNbByTagSorted.items() :
            label = labelError[ 0 ]
            percentage = labelError[ 1 ] / nbSamples
            mapErrorPercentNbByTag[ label ] = "{0:.0f}%".format( percentage * 100 )
    
        # Sort by value
        mapErrorNbPercentByTagSorted = \
            OrderedDict(
                sorted( mapErrorPercentNbByTag.items(), key=lambda t: t[1], reverse=True )
        )
    
        ## Error repartition by label
        print( "% errors by tag for", key, ": ", mapErrorPercentNbByTag )
    
        ## Graph
        if ( show_plot ) :
            x = np.arange( len( mapErrorNbByTagSorted ) )
            plt.bar( x, mapErrorNbByTagSorted.values() )
            plt.xticks( x, mapErrorNbByTagSorted.keys( ) )
            plt.title( "Error repartition for " + key )
            plt.show()
        
        return mapErrorNbByTagSorted, mapErrorPercentNbByTag

