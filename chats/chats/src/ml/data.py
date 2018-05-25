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
    
class DataSet() :
    "Data set"
    def __init__( self, X, Y, imgPath, tag = None, weight = None ):
        self.X = X
        self.Y = Y
        self.imgPath = imgPath
        self.tag = tag;
        self.weight = weight
            