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

# For debug
debugUseScreen = True
debugIdConconfig = 1

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

def testDataSource() :
    import tensorflow as tf
    import numpy as np

    # making fake data using numpy
#    train_data = ( np.random.sample( (10,2) ), np.random.sample( (10,1) ))
#    test_data = (np.random.sample((5,2)), np.random.sample((5,1)))

    train_data = {
        "X": np.array( [ [ 1.0, 2.0 ], [ 3.0, 4.0 ], [ 5.0, 6.0 ] ] ) ,
        "Y": np.array( [ [ 1.0 ], [ 3.0 ], [ 5.0 ] ] )
    }
    test_data =  {
        "X": np.array( [ [ 1.0, 2.0 ], [ 3.0, 4.0 ] ] ),
        "Y": np.array( [ [ 1.0 ], [ 3.0 ] ] )
    }
 
    catDs = cats.CatNormalizedDataSource( { "pxWidth": 64 } )
    catDs.setImageWidth( 64 )
    ( dataTrn, dataDev, _ ) = catDs.getDatasets( False )
 
    train_data = { "X": dataTrn.X, "Y": dataTrn.Y }
    test_data  = { "X": dataDev.X, "Y": dataDev.Y }
 
    print( "Numpy X shape :", train_data[ "X" ].shape )
    print( "Numpy X type  :", train_data[ "X" ].dtype )
    print( "Numpy Y shape :", train_data[ "Y" ].shape )
    print( "Numpy Y type  :", train_data[ "Y" ].dtype )
 
    datasetTrn = tf.data.Dataset.from_tensor_slices( train_data ).batch( 2 )
    datasetDev = tf.data.Dataset.from_tensor_slices( test_data ).shuffle( 100 ).batch( 2 )
 
    train_iterator = datasetTrn.make_initializable_iterator()
    test_iterator  = datasetDev.make_initializable_iterator()
 
    print( "train_iterator.output_shapes:", train_iterator.output_shapes )
    print( "train_iterator.output_types:", train_iterator.output_types )
 
 
    handle = tf.placeholder(tf.string, shape=[], name="Dataset_placeholder" )
    print( "handle shape:", handle.get_shape() )
 
    X_shape = datasetTrn.output_shapes[ "X" ]
    Y_shape = datasetTrn.output_shapes[ "Y" ]
     
    X_type = datasetTrn.output_types[ "X" ]
    Y_type = datasetTrn.output_types[ "Y" ]

    print( "dataset x shape:", X_shape )
     
    theIter = tf.data.Iterator.from_string_handle(
        handle, 
        output_types={'X': X_type, 'Y': Y_type },
        # {'X': TensorShape([Dimension(None), Dimension(2)]), 'Y': TensorShape([Dimension(None), Dimension(1)])}
        output_shapes={ "X": X_shape, "Y": Y_shape }
    )
 
    x = theIter.get_next()[ "X" ]
    print( "x shape:", x.shape )
 
    y_hat = x + 1
 
    with tf.Session() as sess:
 
        init = tf.global_variables_initializer()
        sess.run( init )
 
        # initialise iterators.
        sess.run( train_iterator.initializer )
        sess.run( test_iterator.initializer  )
 
        train_handle = sess.run( train_iterator.string_handle() )
        test_handle  = sess.run( test_iterator.string_handle() )
 
        print( "TRN:" )
        for i in range( 1 ):
            data = sess.run( x, feed_dict = {handle: train_handle } )
            print( data.shape )
 
        print( "DEV:" )
        for i in range( 1 ):
            data = sess.run( y_hat, feed_dict = {handle: test_handle} )
            print( data.shape )
 
    sys.exit()

    
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
            configDoer   = control.ConfigDoer     ( conn )
            hpDoer       = control.HyperParamsDoer( conn )
            runsDoer     = control.RunsDoer       ( conn )
            startRunDoer = control.StartRunDoer   ( conn, configMachinesForms )

            mainWindow = view.MainWindow( configDoer, hpDoer, runsDoer, startRunDoer )
            ( idConfig, buttonClicked, runParams ) = mainWindow.showAndSelectConf( configs )
        else :
            ( idConfig, buttonClicked, runParams ) = (
                debugIdConconfig,
                "Train",
                { "comment": "", "tune": False, "showPlots": False, "nbTuning": 2, "isTensorboard": True, "isTensorboardFull": True }
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
        hyperParams = db.getHyperParams( conn, config[ "idHyperParams" ] )

        # Get machine data source
        machineDataSourceClass = configDatasources[ machineName ]
        if ( machineDataSourceClass == None ) :
            raise ValueError( "Unknown machine data source class", machineName )

        dataSource = instantiateClass( machineDataSourceClass, hyperParams )
        # set image width
        dataSource.setImageWidth( config[ "imageSize" ] )

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
            print( "Train machine", machineName )

            comment     = runParams[ "comment" ]
            tune        = runParams[ "tune" ]
            nbTuning    = runParams[ "nbTuning" ]
            showPlots   = runParams[ "showPlots" ]

            # set run params
            ml.setRunParams( runParams )

            ml.train( conn, config, comment, tune = tune, showPlots = showPlots )
