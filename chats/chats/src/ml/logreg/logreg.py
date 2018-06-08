'''
Created on 25 mai 2018

@author: frup82455
'''

from ml.machine import AbstractMachine
from tensorflow.python.summary.writer.writer import FileWriter

class HandMadeLogregMachine( AbstractMachine ):

    def __init__( self, params = None ):
        super().__init__( params )
        
    def getSession( self ):
        "dummy session"
        session = FileWriter( "run/session.txt" )
        return session