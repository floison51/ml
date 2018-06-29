'''
Created on 25 mai 2018
DataSource for cats
@author: frup82455
'''

from ml.data import DataSource, DataSet
import const.constants as const

import os
import h5py
import numpy as np
import sys
from PIL import Image


class CatRawDataSource( DataSource ):
    "Get cats raw data set, read from files"

    def __init__( self, params = None ):
        super().__init__( params )

    def getDatasets( self, isLoadWeights ):

        if ( self.imagePathes != None ) :

            pixes = []

            for imagePath in self.imagePathes :
                # load image
                img = Image.open( imagePath )
                # Resize image
                resizedImg = img.resize( ( self.pxWidth, self.pxWidth ) )
                # populate lists
                pix = np.array( resizedImg )

                pixes.append( pix )

            dev_X = np.array( pixes, dtype = np.float32 )
            
            # Create data sets
            datasetTrn = None
            datasetDev = DataSet( dev_X, dev_X, None, None, self.imagePathes, None )

        else :

            # Load from h5py data sets

            # Base dir for cats and not cats images
            baseDir = os.getcwd() + "/" + self.path

            trn_dataset = h5py.File( baseDir + "/train_chats-" + str( self.pxWidth ) + ".h5", "r" )
            trn_set_x_orig = np.array(trn_dataset["x"][:]) # your train set features
            trn_set_y_orig = np.array(trn_dataset["y"][:]) # your train set labels

            dev_dataset = h5py.File( baseDir + "/dev_chats-" + str( self.pxWidth ) + ".h5", "r")
            dev_set_x_orig = np.array( dev_dataset["x"][:] ) # your test set features
            dev_set_y_orig = np.array( dev_dataset["y"][:] ) # your test set labels

            trn_set_x = trn_set_x_orig.astype( np.float32 )
            dev_set_x = dev_set_x_orig.astype( np.float32 )

            # passer de (476,) (risque) a  (1,476)
            trn_set_y = trn_set_y_orig.reshape((1, trn_set_y_orig.shape[0]))
            dev_set_y = dev_set_y_orig.reshape((1, dev_set_y_orig.shape[0]))

            ## replace Boolean by (1,0) float values to be consistent with X
            trn_set_y = trn_set_y.astype( np.float32 )
            dev_set_y = dev_set_y.astype( np.float32 )

            # Image tags
            trn_set_tags_orig = np.array(trn_dataset["tags"][:]) # images tags
            dev_set_tags_orig = np.array(dev_dataset["tags"][:]) # images tags
            # passer de (476,) (risque) a  (1,476)
            trn_set_tags      = trn_set_tags_orig.reshape((1, trn_set_tags_orig.shape[0]))
            dev_set_tags      = dev_set_tags_orig.reshape((1, dev_set_tags_orig.shape[0]))

            # Default weight is 1 (int)
            # If weight is loaded, it is a (1,mx)
            trn_set_weights = 1

            if isLoadWeights :

                ## Convert tags to weights
                trn_set_weights   = self.getWeights( trn_set_tags )

            # Image base path
            trn_imgDir = trn_dataset[ "imgDir" ].value.decode( 'utf-8' )
            dev_imgDir = dev_dataset[ "imgDir" ].value.decode( 'utf-8' )

            # Image relative pathes
            trn_imgPathes = np.array( trn_dataset[ "pathes" ][:] )
            dev_imgPathes = np.array( dev_dataset[ "pathes" ][:] )

            # Create data sets
            datasetTrn = DataSet( trn_set_x_orig, trn_set_x, trn_set_y, trn_imgDir, trn_imgPathes, trn_set_tags, trn_set_weights )
            datasetDev = DataSet( dev_set_x_orig, dev_set_x, dev_set_y, dev_imgDir, dev_imgPathes, dev_set_tags )

            # For tensor flow, we need to transpose data
            self.transpose( datasetTrn )
            self.transpose( datasetDev )

        dataInfo = self.getDataInfo( datasetTrn, datasetDev )

        return ( datasetTrn, datasetDev, dataInfo )

    def getDataInfo( self, datasetTrn, datasetDev ) :
        # Store data info in a dico

        dataInfo = { 
            const.KEY_IS_SUPPORT_BATCH_STREAMING : False,
            const.KEY_TRN_X_SIZE  : None,
            const.KEY_TRN_X_SHAPE : None,
            const.KEY_TRN_Y_SIZE  : None,
            const.KEY_TRN_Y_SHAPE : None,
            const.KEY_DEV_X_SIZE  : None,
            const.KEY_DEV_X_SHAPE : None,
            const.KEY_DEV_Y_SIZE  : None,
            const.KEY_DEV_Y_SHAPE : None,
        }

        if ( datasetTrn != None ) :
            dataInfo[ const.KEY_TRN_X_SIZE  ] = datasetTrn.X.shape[0]
            dataInfo[ const.KEY_TRN_X_SHAPE ] = datasetTrn.X.shape
            dataInfo[ const.KEY_TRN_Y_SIZE  ] = datasetTrn.Y.shape[0]
            dataInfo[ const.KEY_TRN_Y_SHAPE ] = datasetTrn.Y.shape
        
        if ( datasetDev != None ) :
            
            dataInfo[ const.KEY_DEV_X_SIZE  ] = datasetDev.X.shape[0]
            dataInfo[ const.KEY_DEV_X_SHAPE ] = datasetDev.X.shape

            if ( datasetDev.Y is None ) :
                dataInfo[ const.KEY_DEV_Y_SIZE  ] = datasetDev.Y.shape[0]
                dataInfo[ const.KEY_DEV_Y_SHAPE ] = datasetDev.Y.shape

        return dataInfo

    def getWeights( self, tags ):

        weights = []

        for n_tag in tags[ 0 ] :

            tag = n_tag.value.decode( 'utf-8' )

            weight = 1
            if ( tag == "chats" ) :
                weight = 100
            elif ( tag == "chiens" ) :
                weight = -100
            elif ( tag == "loups" ) :
                weight = -100
            elif ( tag == "velos" ) :
                weight = 1
            elif ( tag == "gens" ) :
                weight = 1
            elif ( tag == "fleurs" ) :
                weight = 1
            elif ( tag == "villes" ) :
                weight = 1
            elif ( tag == "voitures" ) :
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

class CatNormalizedDataSource( CatRawDataSource ):
    "Get cats row data set, flatten and normalize it"

    def __init__( self, params = None ):
        super().__init__( params )

    def getDatasets( self, isLoadWeights ):
        # ancestor
        ( datasetTrn, datasetDev, dataInfo ) = super().getDatasets( isLoadWeights )
        
        # normalizetrnDataSet X
        if ( datasetTrn != None ) :
            datasetTrn.X = self.normalize( datasetTrn.X )
            
        if ( datasetDev != None ) :
            datasetDev.X = self.normalize( datasetDev.X )

        return ( datasetTrn, datasetDev, dataInfo )

class CatFlattenNormalizedDataSource( CatNormalizedDataSource ):
    "Get cats row data set, flatten and normalize it"

    def __init__( self, params = None ):
        super().__init__( params )

    def getDatasets( self, isLoadWeights ):
        # ancestor
        ( datasetTrn, datasetDev, _ ) = super().getDatasets( isLoadWeights )

        # flatten data
        datasetTrn.X = self.flatten( datasetTrn.X )
        datasetDev.X = self.flatten( datasetDev.X )

        # re-compute data info because shape changed
        dataInfo = self.getDataInfo( datasetTrn, datasetDev )

        return ( datasetTrn, datasetDev, dataInfo )

