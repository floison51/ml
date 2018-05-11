import db.db as db
import view.view as view
import control.control as control

from tkinter import *

if __name__ == '__main__':

    DB_DIR = "C:/Users/frup82455/git/ml/chats/chats/run/db/chats"
    APP_KEY = "chats"

        # Init DB
    print( "Db dir:", DB_DIR )
    
    with db.initDb( APP_KEY, DB_DIR ) as conn :

        # Read configurations
        configs = db.getConfigsWithMaxDevAccuracy( conn )

        configDoer = control.ConfigDoer( conn )
        
        view.showMainWindow( configs, configDoer )
