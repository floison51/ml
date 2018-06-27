'''
Created on 25 mai 2018
Data Source
@author: frup82455
'''

class DataSource():
    '''
    classdocs
    '''

    def __init__( self, params = None ):
        '''
        Constructor
        '''
        self.params = params
    
    def setImagePathes( self, imagePathes ):
        self.imagePathes = imagePathes
        
    def setImageWidth( self, nbPixels ):
        self.pxWidth = nbPixels
        
    def getDatasets( self, params ): 
        "return ( train test set, dev test set) "
        # Abstract implementation
        return ( None, None )
    
    def flatten( self, z ):
        result = z.reshape( z.shape[0], -1 )
        return result
    
    
    def normalize( self, z ):
        result = z / 255
        return result
    
    def transpose( self, dataset ):
        "Transpose dataset for tensorflow"

        #
        dataset.Y = dataset.Y.T
    
        # Transpose for tensorflow
        if ( not ( dataset.imgPathes is None ) ) :
            dataset.imgPathes = dataset.imgPathes.T
    
        if ( not ( dataset.tags is None ) ) :
            dataset.tags = dataset.tags.T
    
        if ( not ( dataset.weights is None ) ) :
            if ( type( dataset.weights ) != int ) :
                dataset.weights = dataset.weights.T
        
        
class DataSet() :
    "Data set"
    def __init__( self, X_ori, X, Y, imgDir, imgPathes, tags = None, weights = None ):
        self.X_ori = X_ori
        self.X = X
        self.Y = Y
        
        self.imgDir = imgDir
        self.imgPathes = imgPathes
        
        self.tags = tags;
        self.weights = weights
        self.isSupportBatchStreaming = False
           