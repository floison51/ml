'''
Created on 25 mai 2018
DataSource for cats
@author: frup82455
'''

from ml.data import DataSource, DataSet
from ml.machine import Machine

import os
import h5py
import numpy as np
import sys

class CatDataSource( DataSource ):
    "Get cats data set, read from files"

    def __init__( self, params = None ):
        super( CatDataSource, self ).__init__( params )
        
    def getDatasets( self, isLoadWeights ):
    
        # Base dir for cats and not cats images
        baseDir = os.getcwd()
    
        train_dataset = h5py.File( baseDir + '/data/prepared/train_signs.h5', "r")
        train_set_x_orig = np.array(train_dataset["x"][:]) # your train set features
        train_set_y_orig = np.array(train_dataset["y"][:]) # your train set labels
    
        dev_dataset = h5py.File( baseDir + '/data/prepared/dev_signs.h5', "r")
        dev_set_x_orig = np.array( dev_dataset["x"][:] ) # your test set features
        dev_set_y_orig = np.array( dev_dataset["y"][:] ) # your test set labels
    
        train_set_x = train_set_x_orig
        dev_set_x   = dev_set_x_orig
    
        # passer de (476,) (risque) a  (1,476)
        train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        dev_set_y   = dev_set_y_orig.reshape((1, dev_set_y_orig.shape[0]))
    
        ## replace Boolean by (1,0) values
        train_set_y = train_set_y.astype( int )
        dev_set_y   = dev_set_y.astype( int )
    
        # Image tags
        train_set_tag_orig = np.array(train_dataset["tag"][:]) # images tags
        dev_set_tag_orig   = np.array(dev_dataset["tag"][:]) # images tags
        # passer de (476,) (risque) a  (1,476)
        train_set_tag      = train_set_tag_orig.reshape((1, train_set_tag_orig.shape[0]))
        dev_set_tag        = dev_set_tag_orig.reshape((1, dev_set_tag_orig.shape[0]))
    
        # Default weight is 1 (int)
        # If weight is loaded, it is a (1,mx)
        train_set_weight = 1
    
        if isLoadWeights :
    
            ## Convert tags to weights
            train_set_weight   = self.getWeights( train_set_tag )
    
        # Image relative pathes
        train_imgPath = np.array( train_dataset[ "path" ][:] )
        dev_imgPath   = np.array( dev_dataset  [ "path" ][:] )
    
        # Create data sets
        trainDataSet = DataSet( train_set_x, train_set_y, train_imgPath, train_set_tag, train_set_weight )
        devDataSet   = DataSet( dev_set_x  , dev_set_y  , dev_imgPath  , dev_set_tag )
        
        return ( trainDataSet, devDataSet )

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

        