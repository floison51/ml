'''
Created on 5 juin 2018

@author: LOISON
'''
from ml.cats.cats import CatNormalizedDataSource

import tensorflow as tf
import const

class TensorFlowDataSource( CatNormalizedDataSource ):
    '''
    classdocs
    '''


    def __init__( self, params ):
        super().__init__( params )

    def isSupportBatchStreaming( self ) :
        return True

    def getDatasets( self, isLoadWeights ):
        # ancestor
        ( trnDataSet_numpy, devDataSet_numpy, dataInfo ) = super().getDatasets( isLoadWeights )

        # get batch size
        self.batchSize = self.params[ "hyperParameters" ][ const.constants.KEY_MINIBATCH_SIZE ]

        # tensor flow sliced data source line = [ data ]
        trnDataSet = tf.data.Dataset.from_tensor_slices( { 
            "X" : trnDataSet_numpy.X, 
            "Y" : trnDataSet_numpy.Y,
        } )
        trnDataSet = trnDataSet.batch( self.batchSize )

        devDataSet = tf.data.Dataset.from_tensor_slices( { "X" : devDataSet_numpy.X, "Y" : devDataSet_numpy.Y } )
        devDataSet = devDataSet.batch( self.batchSize )

        dataInfo[ const.constants.KEY_IS_SUPPORT_BATCH_STREAMING ] = True
        
        return ( trnDataSet, devDataSet, dataInfo )
