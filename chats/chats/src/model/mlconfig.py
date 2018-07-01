'''
Created on 27 mai 2018

@author: fran
'''

from db.db import getHyperParams

class MlConfig( dict ):
    "Machine Leaning configuration"

    def getHyperParams( self, conn, dataset ) :

        # Get hyper params values
        hyperParams = getHyperParams( conn, dataset[ "id" ], self["id" ] )

        return hyperParams
