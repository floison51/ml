import db.db as db
import view.view as view
import control.control as control
import os
from six.moves import configparser
import sys
import importlib

from tkinter import *

import const.constants as const
from ml.machine import Machine
from ml.data import DataSource, DataSet

# A effacer
import ml.cats.cats as cats
import ml.logreg.logreg as logreg

def updateMachines( conn ):
    
    # Read init file
    configMachines = configparser.ConfigParser()
    # Leave keys unchanged
    configMachines.optionxform = str
    configMachines.read( "machines.ini" )
    
    # get machines section
    machines = configMachines.items( "Classes" )
    
    for machine_class in machines :
        # update db
        db.addUniqueMachineName( conn, machine_class[ 0 ] )
    
    # commit result
    conn.commit()
    
    return configMachines
    
if __name__ == '__main__':

    DB_DIR = os.getcwd().replace( "\\", "/" ) + "/run/db/chats"
    APP_KEY = "chats"

        # Init DB
    print( "Db dir:", DB_DIR )

    with db.initDb( APP_KEY, DB_DIR ) as conn :

        # update machines
        configMachines = updateMachines( conn )
        
        # Read configurations
        configs = db.getConfigsWithMaxDevAccuracy( conn )

        configDoer  = control.ConfigDoer     ( conn )
        hpDoer      = control.HyperParamsDoer( conn )
        runsDoer    = control.RunsDoer( conn )

        mainWindow = view.MainWindow( configDoer, hpDoer, runsDoer )
        ( idConfig, buttonClicked ) = mainWindow.showAndSelectConf( configs )

        # cancel?
        if ( buttonClicked == "Cancel" ) :
            print( "Operation cancelled by user" )
            sys.exit( 10 )
            
        # Read config
        config = db.getConfig( conn, idConfig );
        # get machine name
        machineName = db.getMachineNameById( conn, config[ "idMachine" ] )
        
        # TODO write machine main params
        #print( "Structure:", structure )

        # TODO : inistiate actual ML from conf
        
        # Get machine data source
        machineDataSourceClass = configMachines.get( "DataSources", machineName )
        if ( machineDataSourceClass == None ) :
            raise ValueError( "Unknown machine data source", machineName )
        
        module_name, class_name = machineDataSourceClass.rsplit(".", 1)
        MyClass = getattr( importlib.import_module(module_name), class_name)
        instance = MyClass()
        print( instance )
        
        dataSource = cats.CatDataSource()
        ml = logreg.HandMadeLogregMachine()

        # Print system info
        ml.printSystemInfo()

        # Load data
        ( datasetTrn, datasetDev ) = dataSource.getDatasets( isLoadWeights = False );

        # Save original data
        ( X_ori, Y_ori ) = ( datasetTrn.X, datasetDev.Y )

        # flatten data
        datasetTrn.X = dataSource.flatten( datasetTrn.X )
        datasetDev.X = dataSource.flatten( datasetDev.X )

        # normalize X
        datasetTrn.X = dataSource.normalize( datasetTrn.X )
        datasetDev.X = dataSource.normalize( datasetDev.X )

        # Store data info in a dico 
        dataInfo = {
            const.KEY_TRN_X_SIZE    : str( datasetTrn.X.shape[1] ),
            const.KEY_TRN_X_SHAPE   : str( datasetTrn.X.shape ),
            const.KEY_TRN_Y_SHAPE   : str( datasetTrn.Y.shape ),
            const.KEY_DEV_X_SIZE    : str( datasetDev.X.shape[1] ),
            const.KEY_DEV_X_SHAPE   : str( datasetDev.X.shape ),
            const.KEY_DEV_Y_SHAPE   : str( datasetDev.Y.shape ),
        }
        
        print()
        print ("number of training examples = " + str( dataInfo[ const.KEY_TRN_X_SIZE ] ) )
        print ("number of dev test examples = " + str( dataInfo[ const.KEY_DEV_X_SIZE ] ) )
        print ("X_train shape: " + str( dataInfo[ const.KEY_TRN_X_SHAPE ] ) )
        print ("Y_train shape: " + str( dataInfo[ const.KEY_TRN_Y_SHAPE ] ) )
        print ("X_test shape: "  + str( dataInfo[ const.KEY_DEV_X_SHAPE ] ) )
        print ("Y_test shape: "  + str( dataInfo[ const.KEY_DEV_Y_SHAPE ] ) )
        print ()

