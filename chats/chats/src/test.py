import db.db as db
import view.view as view
import control.control as control
import os

from tkinter import *

from ml.machine import Machine
from ml.data import DataSource, DataSet

# A effacer
import ml.cats.cats as cats
import ml.logreg.logreg as logreg

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
        #idConf = mainWindow.showAndSelectConf( configs )

        # TODO write machine main params
        #print( "Structure:", structure )

        # TODO : inistiate actual ML from conf
        dataSource = cats.CatDataSource()
        ml = logreg.HandMadeLogregMachine()

        # Print system info
        ml.printSystemInfo()

        # Load data
        ( datasetTrain, datasetDev ) = dataSource.getDatasets( isLoadWeights = False );

        # Save original data
        ( X_ori, Y_ori ) = ( datasetTrain.X, datasetDev.Y )

        # flatten data
        datasetTrain.X = dataSource.flatten( datasetTrain.X )
        datasetDev.X   = dataSource.flatten( datasetDev.X )
        
        # normalize X
        datasetTrain.X = dataSource.normalize( datasetTrain.X )
        datasetDev.X   = dataSource.normalize( datasetDev.X )
        
        print()
        print ("number of training examples = " + str( datasetTrain.X.shape[1] ) )
        print ("number of dev test examples = " + str( datasetDev.X.shape[1] ) )
        print ("X_train shape: " + str( datasetTrain.X.shape ) )
        print ("Y_train shape: " + str( datasetTrain.Y.shape ) )
        print ("X_test shape: "  + str( datasetDev.X.shape ) )
        print ("Y_test shape: "  + str( datasetDev.Y.shape ) )
        print ()

