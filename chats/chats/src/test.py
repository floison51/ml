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
from ml.cats import cats
from absl.testing.parameterized import parameters

# For debug
debugUseScreen = False
debugIdConfig  = 1
debugCommand   = "Predict"

DB_DIR = os.getcwd().replace( "\\", "/" ) + "/run/db/chats-debug"
APP_KEY = "chats"

def instantiateClass( classFqName, params ) :
    module_name, class_name = classFqName.rsplit(".", 1)
    TheClass = getattr( importlib.import_module(module_name), class_name)
    instance = TheClass( params )

    return instance


def updateMachines( conn ):

    # Read init file
    iniMachines = configparser.ConfigParser()
    # Leave keys unchanged
    iniMachines.optionxform = str
    iniMachines.read( "machines.ini" )

    # Read init file of forms
    configMachinesForms = configparser.ConfigParser()
    # Leave keys unchanged
    configMachinesForms.optionxform = str
    configMachinesForms.read( "machinesForms.ini" )

    configMachineFormsResult = {}

    # get machines section
    machines = iniMachines.items( "Classes" )

    for machineConf in machines :

        machineName  = machineConf[ 0 ]
        machineClass = machineConf[ 1 ]

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

    # Data sources
    configDatasourceResult = {}

    # get data source section
    dataSourceConfs = iniMachines.items( "DataSources" )

    for dataSourceConf in dataSourceConfs :

        machineName     = dataSourceConf[ 0 ]
        datasourceClass = dataSourceConf[ 1 ]

        # save to conf
        configDatasourceResult[ machineName ] = datasourceClass

    return iniMachines, configDatasourceResult, configMachineFormsResult

def prepareData( dataSource ):

    # Load data
    ( datasetTrn, datasetDev, dataInfo ) = dataSource.getDatasets( isLoadWeights = False );

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

    # Init DB
    print( "Db dir:", DB_DIR )

    with db.initDb( APP_KEY, DB_DIR ) as conn :

        # test (debug)
        #db.test( conn )

        # update machines
        ( iniMachines, configDatasources, configMachinesForms ) = updateMachines( conn )

        # Read configurations
        configs = db.getConfigsWithMaxDevAccuracy( conn )

        if ( debugUseScreen ) :
            configDoer     = control.ConfigDoer      ( conn )
            hpDoer         = control.HyperParamsDoer ( conn )
            runsDoer       = control.RunsDoer        ( conn )
            startRunDoer   = control.StartRunDoer    ( conn, configMachinesForms )
            predictRunDoer = control.StartPredictDoer( conn )

            mainWindow = view.MainWindow( configDoer, hpDoer, runsDoer, startRunDoer, predictRunDoer )
            ( idConfig, buttonClicked, runParams, predictParams ) = mainWindow.showAndSelectConf( configs )
        else :
            ( idConfig, buttonClicked, runParams, predictParams ) = (
                debugIdConfig,
                debugCommand,
                { "comment": "", "tune": False, "showPlots": False, "nbTuning": 2, "isTensorboard": True, "isTensorboardFull": False },
                { "choiceHyperParams" : 1, "choiceData" : 1 }
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

        # get hyper parameters
        if ( buttonClicked == "Train" ) :

            # Get config hyper parameters
            hyperParams = db.getHyperParams( conn, config[ "idHyperParams" ] )

        elif ( buttonClicked == "Predict" ) :

            # hyper parameters depend on choice
            choiceHp = predictParams[ "choiceHyperParams" ]

            if ( choiceHp == 1 ) :

                # Config hyper params
                hyperParams = db.getHyperParams( conn, config[ "idHyperParams" ] )

                # Last idRun
                idRun = db.getRunIdLast( conn, config[ "id" ] )

            elif ( choiceHp == 2 ) :

                # Get best hyper parameters
                ( hyperParams, _, idRun ) = db.getBestHyperParams( conn, idConfig )

                # Check run structure and pixel size match with conf
                run = db.getRun( conn, idRun )

                runStructure = None
                if ( run[ "conf_saved_info" ] != None ) :
                    runStructure = run[ "conf_saved_info" ][ "structure" ]

                if ( ( runStructure != None ) and ( config[ "structure" ] != runStructure ) ):
                    raise ValueError( "run 'structure' != config 'structure'" )

                runImageSize = None
                if ( run[ "conf_saved_info" ] != None ) :
                    runImageSize = run[ "conf_saved_info" ][ "imageSize" ]

                if ( ( runImageSize != None ) and ( config[ "imageSize" ] != runImageSize ) ):
                    raise ValueError( "run 'imageSize' != config 'imageSize'" )

            else :
                raise ValueError( "Unknown hyper parameters choice " + choiceHp )
        else :
            raise ValueError( "Unknown action " + buttonClicked )

        # Data source may depend on choice
        if ( ( predictParams != None ) and ( "choiceData" in predictParams ) ) :
            choiceData = predictParams[ "choiceData" ]

        # Get machine data source
        machineDataSourceClass = configDatasources[ machineName ]
        if ( machineDataSourceClass == None ) :
            raise ValueError( "Unknown machine data source class", machineName )

        # Get data
        if ( \
            ( buttonClicked == "Train" ) or \
            ( ( buttonClicked == "Predict" ) and ( choiceData == 1 ) ) \
        ) :
            dataSource = instantiateClass( machineDataSourceClass, hyperParams )
            # set image width
            dataSource.setImageWidth( config[ "imageSize" ] )
        else :
            raise ValueError( "DataSource case not yet supported" )

        # Get machine class
        machineClass = iniMachines.get( "Classes", machineName )
        if ( machineClass == None ) :
            raise ValueError( "Unknown machine class", machineClass )

        ml = instantiateClass( machineClass, None )

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

            # set run params
            ml.setRunParams( runParams )

            showPlots   = runParams[ "showPlots" ]
            tune        = runParams[ "tune" ]
            nbTuning    = runParams[ "nbTuning" ]
            comment     = runParams[ "comment" ]

            print( "Train machine", machineName )
            ml.train( conn, config, comment, tune = tune, showPlots = showPlots )

        elif ( buttonClicked == "Predict" ) :
            print( "Predict from machine", machineName )
            ml.predict( conn, config, idRun )
