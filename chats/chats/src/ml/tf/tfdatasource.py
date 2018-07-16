'''
Created on 5 juin 2018

@author: LOISON
'''
from ml.cats.cats import CatNormalizedDataSource
from ml.data import DataSet

import tensorflow as tf
import const

import os
import h5py
import numpy as np

from PIL import Image

class TensorFlowDataSource( CatNormalizedDataSource ):
    '''
    classdocs
    '''

    def __init__( self, params ):
        super().__init__( params )

    def isSupportBatchStreaming( self ) :
        return True

    def getDatasets( self, isLoadWeights ):

        if ( self.imagePathes != None ) :

            self.getDatasets( isLoadWeights )

        else :

            # Load from h5py data sets

            # Base dir for cats and not cats images
            baseDirTrn  = os.getcwd() + "/" + self.pathTrn
            baseDirDev  = os.getcwd() + "/" + self.pathDev

            with h5py.File( baseDirTrn + "/train_chats-" + str( self.pxWidth ) + "-tfrecord-metadata.h5", "r" ) as trn_dataset_metadata :
                datasetTrn = self.getDataset( trn_dataset_metadata, isLoadWeights )
                # Path to TFRecord files
                datasetTrn.XY = [ baseDirTrn + "/" + trn_dataset_metadata[ "XY_tfrecordPath" ].value.decode( 'utf-8' ) ]

            with h5py.File( baseDirDev + "/dev_chats-"   + str( self.pxWidth ) + "-tfrecord-metadata.h5", "r" ) as dev_dataset_metadata :
                datasetDev = self.getDataset( dev_dataset_metadata, isLoadWeights )
                # Path to TFRecord files
                datasetDev.XY = [ baseDirDev + "/" + dev_dataset_metadata[ "XY_tfrecordPath" ].value.decode( 'utf-8' ) ]

        dataInfo = self.getDataInfo( datasetTrn, datasetDev )

        # get batch size
        self.batchSize = self.params[ "hyperParameters" ][ const.constants.KEY_MINIBATCH_SIZE ]

        dataInfo[ const.constants.KEY_IS_SUPPORT_BATCH_STREAMING ] = True

        return ( datasetTrn, datasetDev, dataInfo )

    read_features = {
        'X': tf.FixedLenFeature( [], dtype=tf.string ),
        'Y': tf.FixedLenFeature( [], dtype=tf.int64 ),
    }

