'''
Created on 25 mai 2018
DataSource for cats
@author: frup82455
'''

from ml.data import DataSource, DataSet

import os
import h5py
import numpy as np
import sys

class CatRawDataSource( DataSource ):
    "Get cats raw data set, read from files"

    def __init__( self, params = None ):
        super( CatRawDataSource, self ).__init__( params )
        
    def getDatasets( self, isLoadWeights ):
    
        # Base dir for cats and not cats images
        baseDir = os.getcwd()
    
        trn_dataset = h5py.File( baseDir + "/data/prepared/train_chats-" + str( self.pxWidth ) + ".h5", "r" )
        trn_set_x_orig = np.array(trn_dataset["x"][:]) # your train set features
        trn_set_y_orig = np.array(trn_dataset["y"][:]) # your train set labels
    
        dev_dataset = h5py.File( baseDir + "/data/prepared/dev_chats-" + str( self.pxWidth ) + ".h5", "r")
        dev_set_x_orig = np.array( dev_dataset["x"][:] ) # your test set features
        dev_set_y_orig = np.array( dev_dataset["y"][:] ) # your test set labels
    
        trn_set_x = trn_set_x_orig
        dev_set_x = dev_set_x_orig
    
        # passer de (476,) (risque) a  (1,476)
        trn_set_y = trn_set_y_orig.reshape((1, trn_set_y_orig.shape[0]))
        dev_set_y = dev_set_y_orig.reshape((1, dev_set_y_orig.shape[0]))
    
        ## replace Boolean by (1,0) values
        trn_set_y = trn_set_y.astype( int )
        dev_set_y = dev_set_y.astype( int )
    
        # Image tags
        trn_set_tag_orig = np.array(trn_dataset["tag"][:]) # images tags
        dev_set_tag_orig = np.array(dev_dataset["tag"][:]) # images tags
        # passer de (476,) (risque) a  (1,476)
        trn_set_tag      = trn_set_tag_orig.reshape((1, trn_set_tag_orig.shape[0]))
        dev_set_tag      = dev_set_tag_orig.reshape((1, dev_set_tag_orig.shape[0]))
    
        # Default weight is 1 (int)
        # If weight is loaded, it is a (1,mx)
        trn_set_weight = 1
    
        if isLoadWeights :
    
            ## Convert tags to weights
            trn_set_weight   = self.getWeights( trn_set_tag )
    
        # Image relative pathes
        trn_imgPath = np.array( trn_dataset[ "path" ][:] )
        dev_imgPath = np.array( dev_dataset  [ "path" ][:] )
    
        # Create data sets
        trnDataSet = DataSet( trn_set_x_orig, trn_set_x, trn_set_y, trn_imgPath, trn_set_tag, trn_set_weight )
        devDataSet = DataSet( dev_set_x_orig, dev_set_x  , dev_set_y  , dev_imgPath  , dev_set_tag )
        
        # For tensor flow, we need to transpose data
        self.transpose( trnDataSet )
        self.transpose( devDataSet )
        
        return ( trnDataSet, devDataSet )

    def getWeights( self, tags ):
    
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

class CatFlattenNormalizedDataSource( CatRawDataSource ):
    "Get cats row data set, flatten and normalize it"

    def __init__( self, params = None ):
        super( CatRawDataSource, self ).__init__( params )
        
    def getDatasets( self, isLoadWeights ):
        # ancestor
        ( datasetTrn, datasetDev ) = super().getDatasets( isLoadWeights )
        
        # flatten data 
        datasetTrn.X = self.flatten( datasetTrn.X )
        datasetDev.X = self.flatten( datasetDev.X )
    
        # normalize X
        datasetTrn.X = self.normalize( datasetTrn.X )
        datasetDev.X = self.normalize( datasetDev.X )

        return ( datasetTrn, datasetDev )
        
        