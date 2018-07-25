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
        self.imagePathes = None
        self.pathHome = None
        self.pathTrn = None
        self.pathDev = None
        self.inMemory = False
    
    def setInMemory( self, inMemory ):
        self.inMemory = inMemory
        
    def setPathHome( self, path ):
        self.pathHome = path
        
    def setPathTrn( self, path ):
        self.pathTrn = path
        
    def setPathDev( self, path ):
        self.pathDev = path
        
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
        # Nothing to do
        pass
    
        
class DataSet() :
    "Data set"
    def __init__( self, nbSamples, shape_X, shape_Y, X_ori, X, Y, XY, dataHome, imgDir, imgPathes, tags = None, weights = None, inMemory = True ):
        
        self.nbSamples = nbSamples
        self.shape_X = shape_X
        self.shape_Y = shape_Y
        
        self.X_ori = X_ori
        self.X = X
        self.Y = Y
        self.XY = XY
        
        self.dataHome = dataHome
        self.imgDir = imgDir
        self.imgPathes = imgPathes
        
        self.tags = tags;
        self.weights = weights
        
        self.inMemory = inMemory
        