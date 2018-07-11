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

    def getDataset( self, dataset_metadata, isLoadWeights ) :

        # Base dir for cats and not cats images
        baseDirHome = os.getcwd() + "/" + self.pathHome

        # Image base path
        imgDir = dataset_metadata.attrs[ "imgDir" ].decode( 'utf-8' )
        
        # Nb samples
        nbSamples = int( dataset_metadata.attrs[ "nbSamples" ] )
        # Shapes
        shape_X   = dataset_metadata.attrs[ "shape_X" ].tolist()
        shape_Y   = dataset_metadata.attrs[ "shape_Y" ].tolist()

        x_orig = dataset_metadata.get( "X" )
        x = None

        if ( x_orig is not None ) :
            x = x_orig.astype( np.float32 )

        y = dataset_metadata.get( "Y" )
        if ( y is not None ) :

            y = np.array( dataset_metadata[ "Y" ][:]) # your train set labels
            y = y.astype( np.float32 )

            # passer de (476,) (risque) a  (1,476)
            y = y.reshape( ( 1, y.shape[ 0 ] ) )
            ## replace Boolean by (1,0) float values to be consistent with X
            y = y.astype( np.float32 )

        # Image tags
        tags_orig = np.array( dataset_metadata[ "tags" ][:] ) # images tags
        # passer de (476,) (risque) a  (1,476)
        tags      = tags_orig.reshape( ( 1, tags_orig.shape[0] ) )

        # Default weight is 1 (int)
        # If weight is loaded, it is a (1,mx)
        weights = 1

        if isLoadWeights :

            ## Convert tags to weights
            weights   = self.getWeights( tags )

        # Image relative pathes
        imgPathes = np.array( dataset_metadata[ "pathes" ][:] )

        # Create data sets
        dataset = DataSet( nbSamples, shape_X, shape_Y, x_orig, x, y, baseDirHome, imgDir, imgPathes, tags, weights )

        return dataset

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
            baseDirTrn  = os.getcwd() + "/" + self.pathTrn
            baseDirDev  = os.getcwd() + "/" + self.pathDev

            with h5py.File( baseDirTrn + "/train_chats-" + str( self.pxWidth ) + ".h5", "r" ) as trn_dataset_metadata :
                datasetTrn = self.super().getDataset( trn_dataset_metadata )


            with h5py.File( baseDirDev + "/dev_chats-" + str( self.pxWidth ) + ".h5", "r") as dev_dataset_metadata :
                datasetDev = self.super().getDataset( dev_dataset_metadata )

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
            dataInfo[ const.KEY_TRN_X_SIZE  ] = datasetTrn.nbSamples
            dataInfo[ const.KEY_TRN_X_SHAPE ] = datasetTrn.shape_X
            dataInfo[ const.KEY_TRN_Y_SIZE  ] = datasetTrn.nbSamples
            dataInfo[ const.KEY_TRN_Y_SHAPE ] = datasetTrn.shape_Y

        if ( datasetDev != None ) :

            dataInfo[ const.KEY_DEV_X_SIZE  ] = datasetDev.nbSamples
            dataInfo[ const.KEY_DEV_X_SHAPE ] = datasetDev.shape_X

            if ( not ( datasetDev.Y is None ) ) :
                dataInfo[ const.KEY_DEV_Y_SIZE  ] = datasetDev.nbSamples
                dataInfo[ const.KEY_DEV_Y_SHAPE ] = datasetDev.shape_Y

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

