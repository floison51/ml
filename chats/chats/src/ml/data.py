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
    
    def setImageWidth( self, nbPixels ):
        self.pxWidth = nbPixels
        
    def getDatasets( self, params ): 
        "return ( train test set, dev test set) "
        # Abstract implementation
        return ( None, None )
    
    def flatten( self, z ):
        result = z.reshape( z.shape[0], -1 ).T
        return result
    
    
    def normalize( self, z ):
        result = z / 255
        return result
    
    def transpose( self, dataset ):
        "Transpose dataset for tensorflow"

        dataset.X = dataset.X.T
        dataset.Y = dataset.Y.T
    
        # Transpose for tensorflow
        if ( not ( dataset.imgPath is None ) ) :
            dataset.imgPath = dataset.imgPath.T
    
        if ( not ( dataset.tag is None ) ) :
            dataset.tag = dataset.tag.T
    
        if ( not ( dataset.weight is None ) ) :
            if ( type( dataset.weight ) != int ) :
                dataset.weight = dataset.weight.T
        
        
class DataSet() :
    "Data set"
    def __init__( self, X_ori, X, Y, imgPath, tag = None, weight = None ):
        self.X_ori = X_ori
        self.X = X
        self.Y = Y
        self.imgPath = imgPath
        self.tag = tag;
        self.weight = weight
            