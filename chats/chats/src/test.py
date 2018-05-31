import db.db as db
import view.view as view
import control.control as control
import os
from six.moves import configparser
import sys
import importlib
import socket
import platform

from tkinter import *

import const.constants as const
from ml.machine import AbstractMachine
from ml.data import DataSource, DataSet

def instantiateClass( classFqName ) :
    module_name, class_name = classFqName.rsplit(".", 1)
    TheClass = getattr( importlib.import_module(module_name), class_name)
    instance = TheClass()

    return instance


def updateMachines( conn ):

    # Read init file
    configMachines = configparser.ConfigParser()
    # Leave keys unchanged
    configMachines.optionxform = str
    configMachines.read( "machines.ini" )

    # Read init file of forms
    configMachinesForms = configparser.ConfigParser()
    # Leave keys unchanged
    configMachinesForms.optionxform = str
    configMachinesForms.read( "machinesForms.ini" )

    configMachineFormsResult = {}

    # get machines section
    machines = configMachines.items( "Classes" )

    for machine_class in machines :

        machineName  = machine_class[ 0 ]
        machineClass = machine_class[ 1 ]

        # update db
        db.addUniqueMachineName( conn, machineName )

        # get forms
        forms = configMachinesForms.items( machineClass )
        dicoFields = {}
        for form in forms :
            key = form[ 0 ]
            field = eval( form[ 1 ] )
            dicoFields[ key ] = field

        # register forms by machine name
        configMachineFormsResult[ machineName ] = dicoFields

    # commit result
    conn.commit()

    return configMachines, configMachineFormsResult

def prepareData( dataSource ):
    # Load data
    ( datasetTrn, datasetDev ) = dataSource.getDatasets( isLoadWeights = False );

    # flatten data and transpose for tensorflow
    datasetTrn.X = dataSource.flatten( datasetTrn.X ).T
    datasetDev.X = dataSource.flatten( datasetDev.X ).T

    # normalize X
    datasetTrn.X = dataSource.normalize( datasetTrn.X )
    datasetDev.X = dataSource.normalize( datasetDev.X )

    # transpose Y for tensorflow
    datasetTrn.Y = datasetTrn.Y.T
    datasetDev.Y = datasetDev.Y.T

    # Store data info in a dico
    dataInfo = {
        const.KEY_TRN_X_SIZE    : str( datasetTrn.X.shape[0] ),
        const.KEY_TRN_X_SHAPE   : str( datasetTrn.X.shape ),
        const.KEY_TRN_Y_SHAPE   : str( datasetTrn.Y.shape ),
        const.KEY_DEV_X_SIZE    : str( datasetDev.X.shape[0] ),
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

#     if ( hyperParams[ const.KEY_USE_WEIGHTS ] ) :
#         print ( "  Weights_train shape :", WEIGHT_train.shape )
#     print ()

    return datasetTrn, datasetDev, dataInfo

if __name__ == '__main__':

    DB_DIR = os.getcwd().replace( "\\", "/" ) + "/run/db/chats"
    APP_KEY = "chats"

        # Init DB
    print( "Db dir:", DB_DIR )

    with db.initDb( APP_KEY, DB_DIR ) as conn :

        # test (debug)
        #db.test( conn )

        # update machines
        ( configMachines, confMachinesForms ) = updateMachines( conn )

        # Read configurations
        configs = db.getConfigsWithMaxDevAccuracy( conn )

        configDoer   = control.ConfigDoer     ( conn )
        hpDoer       = control.HyperParamsDoer( conn )
        runsDoer     = control.RunsDoer       ( conn )
        startRunDoer = control.StartRunDoer   ( conn, confMachinesForms )

#         mainWindow = view.MainWindow( configDoer, hpDoer, runsDoer, startRunDoer )
#         ( idConfig, buttonClicked, runParams ) = mainWindow.showAndSelectConf( configs )

        # For debug
        ( idConfig, buttonClicked, runParams ) = (
            3,
            "Train",
            { "comment": "aeff", "tune": False, "showPlots": False, "nbTuning": 2, "isTensorboard": False, "isTensorboardFull": False }
        )

        # cancel?
        if ( buttonClicked == "Cancel" ) :
            print( "Operation cancelled by user" )
            sys.exit( 10 )

        # Read config
        config = db.getConfig( conn, idConfig );
        # get machine name
        machineName = db.getMachineNameById( conn, config[ "idMachine" ] )

        print( "Structure:" )
        print( config[ "structure" ] )

        # Get machine data source
        machineDataSourceClass = configMachines.get( "DataSources", machineName )
        if ( machineDataSourceClass == None ) :
            raise ValueError( "Unknown machine data source class", machineName )

        # Get machine class
        machineClass = configMachines.get( "Classes", machineName )
        if ( machineClass == None ) :
            raise ValueError( "Unknown machine class", machineClass )

        dataSource = instantiateClass( machineDataSourceClass )
        ml = instantiateClass( machineClass )

        # Define system infos
        systemInfos = {}
        hostname = socket.gethostname()
        pythonVersion = sys.version_info[ 0 ]

        systemInfos[ const.KEY_PYTHON_VERSION ] = pythonVersion
        systemInfos[ const.KEY_HOSTNAME ] = hostname
        systemInfos[ const.KEY_OS_NAME ]  = platform.system() + " " + platform.release()
        systemInfos[ const.KEY_TENSOR_FLOW_VERSION ] = hostname

        ml.addSystemInfo( systemInfos )

        # get data
        ( datasetTrn, datasetDev, dataInfos ) = prepareData( dataSource )
        # Set data
        ml.setData( datasetTrn, datasetDev )

        # Set all infos
        ml.setInfos( systemInfos, dataInfos )

        # train?
        if ( buttonClicked == "Train" ) :
            print( "Train machine", machineName )

            comment     = runParams[ "comment" ]
            tune        = runParams[ "tune" ]
            nbTuning    = runParams[ "nbTuning" ]
            showPlots   = runParams[ "showPlots" ]

            # set run params
            ml.setRunParams( runParams )
             
            ml.train( conn, config, comment, tune = tune, showPlots = showPlots )
