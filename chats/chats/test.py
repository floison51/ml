import db.db as db
import view.view as view
import control.control as control
import os

from tkinter import *

if __name__ == '__main__':

    DB_DIR = os.getcwd().replace( "\\", "/" ) + "/run/db/chats"
    APP_KEY = "chats"

        # Init DB
    print( "Db dir:", DB_DIR )
    
    with db.initDb( APP_KEY, DB_DIR ) as conn :

        #db.test( conn )
        
        # Read configurations
        configs = db.getConfigsWithMaxDevAccuracy( conn )

        configDoer  = control.ConfigDoer     ( conn )
        hpDoer      = control.HyperParamsDoer( conn )
        runsDoer    = control.RunsDoer( conn )
        
        mainWindow = view.MainWindow( configDoer, hpDoer, runsDoer )
        mainWindow.show( configs )
