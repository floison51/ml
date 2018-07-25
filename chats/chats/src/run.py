## Logging
import logging
import logging.config

import db.db as db
from view.view import MainWindow
import control.control as control

from six.moves import configparser
import sys
import importlib
import socket
import platform

import const.constants as const

# For debug
debugUseScreen = False
debugDatasetName = "Hand-made original"
debugIdConfig  = 1
debugCommand   = "Train"
debugIsTensorBord=False

## Logging
logging.config.fileConfig( 'logging.conf' )
# create logger
logger = logging.getLogger( 'run' )

logger.info( "Application '" + const.APP_KEY + "', DB dir: '" + const.DB_DIR + "'" )

def instantiateClass( classFqName, params ) :
    module_name, class_name = classFqName.rsplit(".", 1)
    TheClass = getattr( importlib.import_module(module_name), class_name)
    instance = TheClass( params )

    return instance

def updateDatasets( conn, selection ):

    # get selected data set
    selectedDatasetName = selection.get( "selectedDatasetName" )

    # selection OK flag
    selectionOk = False

    # Read init file
    iniDatasets = configparser.ConfigParser()
    # Leave keys unchanged
    iniDatasets.optionxform = str
    iniDatasets.read( "datasets.ini" )

    # Get current
    # Read sections
    for section in iniDatasets.sections() :

        name    = section

        if ( name == selectedDatasetName ) :
            selectionOk = True

        order    = int( iniDatasets.get( section, "order" ) )
        inMemory = iniDatasets.get( section, "inMemory" )
        pathHome = iniDatasets.get( section, "pathHome" )
        pathTrn  = iniDatasets.get( section, "pathTrn" )
        pathDev  = iniDatasets.get( section, "pathDev" )

        # Create dataset
        db.createOrUpdateDataset( conn, name, order, inMemory, pathHome, pathTrn, pathDev )

    # Clear selection if not existing
    if ( not selectionOk ) :
        selection.pop( "selectedDatasetName", None )
        # save selection
        db.setSelection( conn, selection )

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

def strNone( x ):
    if ( x is None ) :
        return "None"
    else :
        return str( x )

def prepareData( dataSource ):

    # Load data
    ( datasetTrn, datasetDev, dataInfo ) = dataSource.getDatasets( isLoadWeights = False );

    logger.info( "" )
    logger.info( "in memory trn dataset  = " + str( datasetTrn.inMemory ) )
    logger.info( "in memory dev dataset  = " + str( datasetDev.inMemory ) )
    logger.info( "number of trn examples = " + strNone( dataInfo[ const.KEY_TRN_X_SIZE ] ) )
    logger.info( "number of dev examples = " + strNone( dataInfo[ const.KEY_DEV_X_SIZE ] ) )
    logger.info( "X_train shape: " + strNone( dataInfo[ const.KEY_TRN_X_SHAPE ] ) )
    logger.info( "Y_train shape: " + strNone( dataInfo[ const.KEY_TRN_Y_SHAPE ] ) )
    logger.info( "X_dev  shape: "  + strNone( dataInfo[ const.KEY_DEV_X_SHAPE ] ) )
    logger.info( "Y_dev  shape: "  + strNone( dataInfo[ const.KEY_DEV_Y_SHAPE ] ) )

#     if ( hyperParams[ const.KEY_USE_WEIGHTS ] ) :
#         print ( "  Weights_train shape :", WEIGHT_train.shape )
#     print ()

    return datasetTrn, datasetDev, dataInfo


if __name__ == '__main__':

    # Init DB
    with db.initDb( const.APP_KEY, const.DB_DIR ) as conn :

        # test (debug)
        #db.test( conn )

        # Read selections
        selection = db.getSelection( conn )

        # update datasets
        updateDatasets( conn, selection )

        # update machines
        ( iniMachines, configDatasources, configMachinesForms ) = updateMachines( conn )

        # Read datasets
        datasets = db.getDatasets( conn )

        if ( debugUseScreen ) :
            configDoer     = control.ConfigDoer      ( conn )
            hpDoer         = control.HyperParamsDoer ( conn )
            runsDoer       = control.RunsDoer        ( conn )
            startRunDoer   = control.StartRunDoer    ( conn, configMachinesForms )
            analyzeDoer    = control.AnalyzeDoer     ( conn )
            predictRunDoer = control.StartPredictDoer( conn )

            mainWindow = MainWindow( configDoer, hpDoer, runsDoer, startRunDoer, analyzeDoer, predictRunDoer )
            ( datasetName, idConfig, buttonClicked, runParams, predictParams ) = mainWindow.showAndSelectConf( conn, datasets, selection )
        else :
            ( datasetName, idConfig, buttonClicked, runParams, predictParams ) = (
                debugDatasetName,
                debugIdConfig,
                debugCommand,
                { "comment": "", "tune": False, "showPlots": False, "nbTuning": 2, "isTensorboard": debugIsTensorBord, "isTensorboardFull": False },
                { "choiceHyperParams" : 1, "choiceData" : 1 }
            )

        # cancel?
        if ( buttonClicked == "Cancel" ) :
            logger.info( "Operation cancelled by user" )
            sys.exit( 10 )

        # dataset
        idDataset = db.getDatasetIdByName( conn, datasetName )
        dataset = db.getDatasetById( conn, idDataset )
        logger.info( "Using dataset {0}".format( dataset ) )

        # Read config
        config = db.getConfig( conn, idConfig );
        # get machine name
        machineName = db.getMachineNameById( conn, config[ "idMachine" ] )

        logger.info( "Structure:" )
        logger.info( config[ "structure" ] )

        # get hyper parameters
        if ( buttonClicked == "Train" ) :

            # Get config hyper parameters
            hyperParams = db.getHyperParams( conn, idDataset, config[ "id" ] )

        elif ( buttonClicked == "Predict" ) :

            # hyper parameters depend on choice
            choiceHp = predictParams[ "choiceHyperParams" ]

            if ( choiceHp == 1 ) :

                # Last idRun
                idRun = db.getRunIdLast( conn, config[ "id" ] )

                # Config hyper params
                run = db.getRun( conn, idRun )
                hyperParams = db.getHyperParamsById( conn, run[ "idHyperParams" ] )

            elif ( choiceHp == 2 ) :

                # Get best hyper parameters
                ( hyperParams, _, idRun ) = db.getBestHyperParams( conn, idDataset, idConfig )

                # Check run structure and pixel size match with conf
                run = db.getRun( conn, idRun )

                runStructure = None
                if ( run[ "conf_saved_info" ] != None ) :
                    runStructure = run[ "conf_saved_info" ][ "structure" ]
                    # trim spaces
                    runStructure = runStructure.strip()

                configStructure = config[ "structure" ].strip()

                if ( ( runStructure != None ) and ( configStructure != runStructure ) ):
                    raise ValueError( "run 'structure' != config 'structure'" )

                runImageSize = None
                if ( run[ "conf_saved_info" ] != None ) :
                    runImageSize = run[ "conf_saved_info" ][ "imageSize" ]

                if ( ( runImageSize != None ) and ( config[ "imageSize" ] != runImageSize ) ):
                    raise ValueError( "run 'imageSize' != config 'imageSize'" )

            else :
                raise ValueError( "Unknown hyper parameters choice " + choiceHp )
        else :
            raise ValueError( "Unknown action " + str( buttonClicked ) )

        # Data source may depend on choice
        choiceData = None

        if ( ( predictParams != None ) and ( "choiceData" in predictParams ) ) :
            choiceData = predictParams[ "choiceData" ]

        # Get machine data source
        machineDataSourceClass = configDatasources[ machineName ]
        if ( machineDataSourceClass == None ) :
            raise ValueError( "Unknown machine data source class", machineName )

        # Get data
        if ( \
            ( buttonClicked == "Train" ) or \
            ( ( buttonClicked == "Predict" ) ) \
        ) :
            dataSource = instantiateClass( machineDataSourceClass, hyperParams )
            # set image width
            dataSource.setImageWidth( config[ "imageSize" ] )

        if ( ( choiceData != None ) and ( choiceData == 2 ) ) :
            # image chosen
            dataSource.setImagePathes( [ predictParams[ "imagePath" ] ] )

        # Tell source inMemory flag
        dataSource.setInMemory( dataset[ "inMemory" ] == "True" )
        
        # Tell source where is data
        dataSource.setPathHome( dataset[ "pathHome" ] )
        dataSource.setPathTrn(  dataset[ "pathTrn" ] )
        dataSource.setPathDev(  dataset[ "pathDev" ] )

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
        ml.setData( dataset[ "id" ], datasetTrn, datasetDev )

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

            logger.info( "Train machine " + machineName )
            ml.train( conn, dataset, config, comment, tune = tune, showPlots = showPlots )

        elif ( buttonClicked == "Predict" ) :
            logger.info( "Predict from machine " + machineName )
            ml.predict( conn, dataset, config, idRun, dataSource.imagePathes )
