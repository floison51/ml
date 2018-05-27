'''
Created on 27 mai 2018

@author: fran
'''

from db.db import getHyperParams

class MlConfig( dict ):
    "Machine Leaning configuration"

    def getHyperParams( self, conn ) :
        
        # get hyper param id
        idHyperParams = self[ "idHyperParams" ]
        
        # Get hyper params values
        hyperParams = getHyperParams( conn, idHyperParams )
        
        return hyperParams