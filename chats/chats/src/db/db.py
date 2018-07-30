'''
Created on 28 avr. 2018
Machine Learning DB
@author: fran
'''

import sqlite3
import os
import datetime
import json

from collections import OrderedDict

import const.constants as const

# Current DB version
DB_VERSION = 1

def getSelection( conn ):

    cursor = conn.cursor();

    try :
        # get existing, if anay
        cursor.execute(
            "select jsonSelection from selections where id=1"
        )

        result = None

        for row in cursor :
            resultJson = row[ 0 ]
            result = json.loads( resultJson )

    finally :
        cursor.close()

    return result

def setSelection( conn, selection ):

    jsonSelection = json.dumps( selection )
    cursor = conn.cursor();

    try :
        # get existing, if anay
        cursor.execute(
            "update selections set jSonSelection=?where id=1",
            ( jsonSelection, )
        )

    finally :
        cursor.close()


def createOrUpdateDataset( conn, name, order, inMemory, pathHome, pathTrn, pathDev ) :

    cursor = conn.cursor();

    try :
        # get existing, if any
        cursor.execute(
            "select id from datasets where name=?" ,
            ( name, )
        )

        # For some reason, cursor.rowcount is NOK, use another method
        idConfig = -1
        for config in cursor:
            idConfig = config[ 0 ]

        if ( idConfig < 0 ) :
            # None, create
            idConfig = createDataset( conn, name, order, inMemory, pathHome, pathTrn, pathDev )

        else :
            # update data set
            updateDataset( conn, idConfig, name, order, inMemory, pathHome, pathTrn, pathDev )

    finally :
        cursor.close()

    return idConfig;

def getDatasetIdByName( conn, name ) :

    cursor = conn.cursor()

    try :
        # get existing, if anay
        cursor.execute(
            "select id from datasets where name=?" ,
            ( name, )
        )

        # For some reason, cursor.rowcount is NOK, use another method
        idResult = -1
        for config in cursor:
            idResult = config[ 0 ]

    finally :
        cursor.close()

    return idResult;

def getDatasetById( conn, idDataset ) :

    cursor = conn.cursor()

    try :
        # get existing, if anay
        cursor.execute(
            "select * from datasets where id=?" ,
            ( idDataset, )
        )

        result = {}

        for row in cursor :
            iCol = 0

            for colName in const.DatasetDico.DISPLAY_FIELDS :
                result[ colName ] = row[ iCol ]
                iCol += 1

    finally :
        cursor.close()

    return result;

def createDataset( conn, name, order, inMemory, pathHome, pathTrn, pathDev ) :

    cursor = conn.cursor()

    try :
        cursor.execute(
            "INSERT INTO datasets VALUES ( null, ?, ?, ?, ?, ?, ? )",
            ( name, order, inMemory, pathHome, pathTrn, pathDev, )
        )

        idResult = cursor.lastrowid

    finally :
        cursor.close();

    return idResult

def updateDataset( conn, idDataset, name, order, inMemory, pathHome, pathTrn, pathDev ) :

    cursor = conn.cursor()

    try :
        cursor.execute(
            "UPDATE datasets set displayOrder=?, name=?, inMemory=?, pathHome=?, pathTrn=?, pathDev=? where id=?",
            ( order, name, inMemory, pathHome, pathTrn, pathDev, idDataset, )
        )

        idResult = cursor.lastrowid

    finally :
        cursor.close();

    return idResult

def getDatasets( conn ) :


    cursor = conn.cursor()

    try :
        # Get data sets
        cursor.execute(
            "select * from datasets order by displayOrder asc"
        )

        results = []

        for row in cursor :

            result = {}

            iCol = 0

            for colName in const.DatasetDico.DISPLAY_FIELDS :
                result[ colName ] = row[ iCol ]
                iCol += 1

            # add result
            results.append( result )

    finally :
        cursor.close();

    return results


def createConfig( conn, idDataset, name, structure, imageSize, machineName, hyper_params ) :

    # Id machine
    idMachine = getIdMachineByName( conn, machineName )

    cursor = conn.cursor()

    try :
        # Create config
        cursor.execute(
            "INSERT INTO configs VALUES ( null, ?, ?, ?, ? )",
            ( name, structure, imageSize, idMachine, )
        )

        idConfig = cursor.lastrowid

        # get hyparams
        idHyperParams = getOrCreateHyperParams( conn, idDataset, idConfig, hyper_params )

        # create selector, no run yet
        createHpRunSelector( conn, idDataset, idConfig, idHyperParams, None )

    finally :
        cursor.close();

    return idConfig

def createHpRunSelector( conn, idDataset, idConfig, idHyperParams, idRun ):
    cursor = conn.cursor()

    try :

        # create selector
        cursor.execute(
            "INSERT INTO hpRunSelector VALUES ( null, ?, ?, ?, ? )",
            ( idDataset, idConfig, idHyperParams, idRun, )
        )

    finally :
        cursor.close();

def updateHpRunSelectorForHp( conn, idDataset, idConfig, idHp ):
    cursor = conn.cursor()

    try :

        cursor.execute(
            "UPDATE hpRunSelector set idHp=? where idDataset=? and idConfig=?",
            ( idHp, idDataset, idConfig, )
        )

    finally :
        cursor.close();

def updateHpRunSelectorForRun( conn, idDataset, idConfig, idRun ):
    cursor = conn.cursor()

    try :

        cursor.execute(
            "UPDATE hpRunSelector set idRun=? where idDataset=? and idConfig=?",
            ( idRun, idDataset, idConfig, )
        )

    finally :
        cursor.close();

def getHyperParamsFrowRow( row ) :

    result = {}

    result[ "id" ]    = row[ 0 ]
    json_hyper_params = row[ 1 ]

    hyper_params = json.loads( json_hyper_params )
    result[ const.KEY_DICO_HYPER_PARAMS ]  = hyper_params

    return result

def getHyperParamsById( conn, idHp ) :

    cursor = conn.cursor()

    try :
        # get hp selector to retrieve idHp
        cursor.execute(
            "select * from hyperparams where id=?",
            ( idHp, )
        )

        result = None

        for row in cursor :
            result = getHyperParamsFrowRow( row )

    finally :
        cursor.close()

    return result


def getHpRunSelector( conn, idDataset, idConfig ) :

    cursor = conn.cursor()
    result = {}

    try :
        # get hp selector to retrieve idHp
        cursor.execute(
            "select * from hpRunSelector where idDataset=? and idConfig=?",
            ( idDataset, idConfig, )
        )

        for row in cursor :
            for i in range( len( const.HpRunSelectorDico.OBJECT_FIELDS ) ) :
                result[ const.HpRunSelectorDico.OBJECT_FIELDS[ i ] ] = row[ i ]

    finally :
        cursor.close();

    return result

def getHyperParams( conn, idDataset, idConfig ) :

    cursor = conn.cursor()

    try :

        # get hp selector to retrieve idHp
        idHyperParams = getSelectedIdHyperparam( conn, idDataset, idConfig )

        # get existing, if any
        cursor.execute(
            "select * from hyperparams where id=?",
            ( idHyperParams, )
        )

        result = {}

        for row in cursor :
            result = getHyperParamsFrowRow( row )

        # resolve datat set name
        dataset = getDatasetById( conn, idDataset )
        result[ const.KEY_DICO_DATASET_NAME ]  = dataset[ "name" ]

    finally :
        cursor.close();

    return result

def getBestHyperParams( conn, idDataSet, idConfig ) :

    cursor = conn.cursor()

    try :
        # get existing, if anay
        cursor.execute(
            "select r.idHyperParams, max(r.dev_accuracy), r.id from configs c, runs r where ( c.id=? and r.idConf=c.id and r.idDataset=? )",
            ( idConfig, idDataSet, )
        )

        devAccuracy = None
        hyperParams = {}
        idRun = None

        idHyperParams = None
        for row in cursor :
            idHyperParams = row[ 0 ]
            devAccuracy   = row[ 1 ]
            idRun         = row[ 2 ]

    finally :
        cursor.close()

    if ( idHyperParams != None ) :
        hyperParams = getHyperParamsById( conn, idHyperParams )

    return ( hyperParams, devAccuracy, idRun )

def getSelectedIdHyperparam( conn, idDataset, idConf ):
    "Get hyper parameter selector given data set and conf"

    selector = getHpRunSelector( conn, idDataset, idConf )

    # For some reason, cursor.rowcount is NOK, use another method
    idHyperparam = None
    if ( selector is not None ) :
        idHyperparam = selector[ "idHp" ]

    return idHyperparam


def getOrCreateHyperParams( conn, hyper_params ) :

    cursor = conn.cursor();

    try :
        # Sort hyperparams by key
        sorted_hyper_params = OrderedDict( sorted( hyper_params.items(), key=lambda t: t[0] ) )
        # JSon conversion
        json_hyper_params   = json.dumps( sorted_hyper_params )

        # get existing, if any
        cursor.execute(
            "select id from hyperparams where json_hyper_params=?",
            ( json_hyper_params, )
        )

        # For some reason, cursor.rowcount is NOK, use another method
        idHpResult = -1
        for ( hp ) in cursor:
            idHpResult = hp[ 0 ]

        if ( idHpResult < 0 ) :
            # None, create
            cursor.execute(
                "INSERT INTO hyperparams VALUES ( null, ? )",
                ( json_hyper_params, )
            )

            idHpResult = cursor.lastrowid

    finally :
        cursor.close()

    return idHpResult;

def getConfig( conn, idConfig ) :

    from model.mlconfig import MlConfig

    cursor = conn.cursor()

    try :
        cursor.execute(
            "select c.* from configs c where c.id=? ",
            ( idConfig, )
        )

        result = MlConfig()

        for row in cursor :

            iCol = 0;

            for colName in const.ConfigsDico.OBJECT_FIELDS :
                result[ colName ] = row[ iCol ]
                iCol += 1

    finally :
        cursor.close();

    return result

def getOrCreateConfig( conn, name, structure, hyperParams ) :

    cursor = conn.cursor();

    try :
        # get existing, if anay
        cursor.execute(
            "select id from configs where name=? and structure=?" ,
            ( name, structure, )
        )

        # For some reason, cursor.rowcount is NOK, use another method
        idResult = -1
        for config in cursor:
            idResult = config[ 0 ]

        if ( idResult < 0 ) :
            # None, create
            idResult = createConfig( conn, name, structure, hyperParams )

    finally :
        cursor.close()

    return idResult;

def updateConfig( conn, config ) :

    cursor = conn.cursor();

    try :
        # Update config
        updateStatement = \
            "update configs set " + \
                 "name=?, " + \
                 "structure=?, " + \
                 "imageSize=?, " + \
                 "idMachine=? " + \
                 "where id=?"

        cursor.execute(
            updateStatement,
            (
                config[ "name" ], config[ "structure" ], config[ "imageSize" ], config[ "idMachine" ], config[ "id" ]
            )
        )

    finally :
        cursor.close();

def deleteConfig( conn, idConf ) :

    cursor = conn.cursor();

    try :
        # Update config
        deleteStatement = "delete from configs where id=?"
        cursor.execute(
            deleteStatement,
            ( idConf, )
        )

    finally :
        cursor.close();

def getConfigsWithMaxDevAccuracy( conn, idDataset, idConfig = None ) :

    from model.mlconfig import MlConfig

    cursor = conn.cursor()

    try :

        parameters = ()

        statement = \
            "select c.id as id, c.name as name, " + \
            "( select m.name from machines m where m.id=c.idMachine ) as machine, " + \
            "c.imageSize as imageSize, c.structure as structure " + \
            "from configs c "

        if ( idConfig != None ) :
            statement += " where c.id=?"
            # Add paremetr
            parameters = parameters + ( idConfig, )

        statement += " order by c.id asc"

        # Update run
        cursor.execute(
            statement,
            parameters
        )

        results = []


        for row in cursor :

            result = MlConfig()

            iCol = 0

            # Object fields
            for colName in const.ConfigsDico.DISPLAY_FIELDS :

                if (
                    ( colName != "bestDevAccuracy" ) and ( colName != "assoTrnAccuracy" ) and
                    ( colName != "idHp" ) and ( colName != "json_hperParams" ) and
                    ( colName != "idLastRun" ) and
                    ( colName != "lastRunDevAccuracy" ) and ( colName != "lastRunAssoTrnAccuracy" )
                ) :
                    result[ colName ] = row[ iCol ]
                    iCol += 1

            # now we have idConf
            curIdConf = result[ "id" ]

            # Append best DEV accuracy and TRN Accuracy
            statementRun = \
                "select r.id from runs r where ( r.idDataset=? and r.idConf=? and r.dev_accuracy=" \
                    "(select max( r2.dev_accuracy) from runs r2 where r2.idDataset=? and r2.idConf=? ) ) order by id desc;"

            # dataset id, config id
            parameters = ( idDataset, curIdConf, idDataset, curIdConf, )

            cursorRun = conn.cursor()

            try :

                cursorRun.execute(
                    statementRun,
                    parameters
                )

                bestDevAccuracy = None
                assoTrnAccuracy = None

                # Get selected run
                for row in cursorRun :

                    idRun = row[ 0 ]
                    run = getRun( conn, idRun )

                    bestDevAccuracy = run[ "dev_accuracy" ]
                    assoTrnAccuracy = run[ "train_accuracy" ]

                    # Read only first row
                    break

                # Add accuracies
                result[ "bestDevAccuracy" ] = bestDevAccuracy
                result[ "assoTrnAccuracy" ] = assoTrnAccuracy

            finally :
                cursorRun.close()

            # idHp and run tin
            result[ "idHp" ] = None
            result[ "json_hperParams" ] = None
            result[ "idLastRun" ] = None
            result[ "lastRunDevAccuracy" ] = None
            result[ "lastRunAssoTrnAccuracy" ] = None

            # get select hp and run
            runSelector = getHpRunSelector( conn, idDataset, curIdConf )
            idHp = runSelector[ "idHp" ]
            result[ "idHp" ]      = idHp
            if ( idHp is not None ) :
                # Get hyper-parameters
                hp = getHyperParamsById( conn, idHp )
                result[ "json_hperParams" ] = hp

            idLastRun = runSelector[ "idRun" ]
            result[ "idLastRun" ] = idLastRun

            if ( idLastRun is not None ) :
                run = getRun( conn, idLastRun )
                result[ "lastRunDevAccuracy" ]     = run[ "dev_accuracy" ]
                result[ "lastRunAssoTrnAccuracy" ] = run[ "train_accuracy" ]

            # add result
            results.append( result )

    finally :
        cursor.close();

    return results

def getMachineNames( conn ) :

    cursor = conn.cursor();

    try :

        # Update run
        cursor.execute(
            "select name from machines"
        )

        result = []

        for row in cursor :
            result.append( row[ 0 ] )

    finally :
        cursor.close();

    return result

def getIdMachineByName( conn, machineName ):

    cursor = conn.cursor();

    try :
        # Update run
        cursor.execute(
            "select id from machines where name=?",
            ( machineName, )
        )

        result = None

        for row in cursor :
            result = row[ 0 ]

    finally :
        cursor.close();

    return result

def getMachineNameById( conn, idMachine ):

    cursor = conn.cursor();

    try :

        # Update run
        cursor.execute(
            "select name from machines where id=?",
            ( idMachine, )
        )

        result = None

        for row in cursor :
            result = row[ 0 ]

    finally :
        cursor.close();

    return result

def getMachineIdByName( conn, machineName ):

    cursor = conn.cursor();

    try :
        # Update run
        cursor.execute(
            "select id from machines where name=?",
            ( machineName, )
        )

        result = None

        for row in cursor :
            result = row[ 0 ]

    finally :
        cursor.close();

    return result

def addUniqueMachineName( conn, machineName ) :

    cursor = conn.cursor()

    try :
        # exists?
        idMachine = getMachineIdByName( conn, machineName )

        if ( idMachine == None ) :
            cursor.execute( '''
                INSERT INTO machines ( name ) VALUES ( ? )''',
                ( machineName, )
            )

            idMachine = cursor.lastrowid

    finally :
        cursor.close();

    return idMachine



def createRun( conn, idDataset, idConfig, runHyperParams ) :

    cursor = conn.cursor()

    try :
        # get config
        config = getConfig( conn, idConfig )

        # Get hyperparams
        idRunHyperParams = getOrCreateHyperParams( conn, runHyperParams )

        # Save conf
        json_conf_saved = json.dumps( config )

        cursor.execute( '''
            INSERT INTO runs ( idDataset, idConf, json_conf_saved, idHyperParams, date ) VALUES ( ?, ?, ?, ?, ? )''',
            ( idDataset, config[ "id" ], json_conf_saved, idRunHyperParams, datetime.datetime.now(), )
        )

        idRun = cursor.lastrowid

    finally :
        cursor.close();

    return idRun

def updateRun( conn, idRun, runHyperParams ) :

    cursor = conn.cursor()

    try :
        # Get hyperparams
        idRunHyperParams = getOrCreateHyperParams( conn, runHyperParams );

        cursor.execute( \
            "update runs set idHyperParams=? where id=?", \
            ( idRunHyperParams, idRun, )
        )

        idRun = cursor.lastrowid

    finally :
        cursor.close();

    return idRun

def updateRunBefore(
    conn, idRun,
    comment = "?",
    system_info = {}, data_info = {},
) :

    cursor = conn.cursor()

    try :
        # JSon conversion
        json_system_info    = json.dumps( system_info )
        json_data_info      = json.dumps( data_info )
        json_perf_info      = json.dumps( {} )
        json_result_info    = json.dumps( {} )

        # Update run
        updateStatement = \
            "update runs set " + \
                 "comment=?, " + \
                 "json_system_info=?, json_data_info=?, json_perf_info=?, json_result_info=?, " + \
                 "perf_index=?, train_accuracy=?, dev_accuracy=? " + \
                 "where id=?"

        cursor.execute(
            updateStatement,
            (
                comment,
                json_system_info, json_data_info, json_perf_info, json_result_info,
                -1, -1, -1,
                idRun,
            )
        )

    finally :
        cursor.close();

def updateRunAfter(
    conn, idRun,
    perf_info = {}, result_info={},
    perf_index = -1, elapsed_second =- 1, train_accuracy = -1, dev_accuracy = -1
) :

    cursor = conn.cursor()

    try :
        # JSon conversion
        json_perf_info      = json.dumps( perf_info )
        json_result_info    = json.dumps( result_info )

        # Update run
        updateStatement = \
            "update runs set " + \
                "json_perf_info=?, json_result_info=?," + \
                 "perf_index=?, elapsed_second=?, train_accuracy=?, dev_accuracy=? " + \
                 "where id=?"

        cursor.execute(
            updateStatement,
            (
                json_perf_info, json_result_info,
                perf_index, elapsed_second, train_accuracy, dev_accuracy,
                idRun,
            )
        )

    finally :
        cursor.close();

    # Save (commit) the changes
    conn.commit()


def getRunFromRow(row):
    # TODO : use dico
    result = {}
    result["id"]                = row[0]
    result["idDataset"]         = row[1]
    result["idConf"]            = row[2]
    result["idHyperParams"]     = row[3]
    result["dateTime"]          = row[4]
    result["comment"]           = row[5]
    result["perf_index"]        = row[6]
    result["elapsed_second"]    = row[7]
    result["train_accuracy"]    = row[8]
    result["dev_accuracy"]      = row[9]
    result["system_info"]       = json.loads(row[10])
    result["data_info"]         = json.loads(row[11])
    result["perf_info"]         = json.loads(row[12])
    result["result_info"]       = json.loads(row[13])
    raw_conf_saved = row[ 14 ]
    if ( raw_conf_saved == None ) :
        result["conf_saved_info"]    = {}
    else :
        result["conf_saved_info"]    = json.loads( raw_conf_saved )

    return result

def getRuns( conn, idDataset, idConf ) :

    cursor = conn.cursor()

    try :
        # Update run
        cursor.execute( '''
            select * from runs where idDataset=? and idConf=?''',
            ( idDataset, idConf, )
        )

        results = []

        # TODO use dico
        for row in cursor :

            result = getRunFromRow( row )

            results.append( result )

    finally :
        cursor.close();

    return results

def getRunIdLast( conn, idConf ) :

    cursor = conn.cursor()

    try :
        # Update run
        cursor.execute( '''
            select max( id ) from runs where idConf=?''',
            (idConf,)
        )

        for row in cursor :

            result = row[ 0 ]

    finally :
        cursor.close();

    return result

def getRun( conn, idRun ) :

    cursor = conn.cursor()

    try :
        # Update run
        cursor.execute( '''
            select * from runs where id=?''',
            (idRun,)
        )

        result = None

        for row in cursor :
            result = getRunFromRow( row )

    finally :
        cursor.close();

    return result

def getDbVersion( cursor ) :

    # Update run
    cursor.execute( "select max( version ) from versions" )

    result = None

    for row in cursor :
        result = row[ 0 ]

    return result

def setDbVersion( cursor, dbVersion ) :

    # Add version
    cursor.execute( "INSERT INTO versions ( version ) VALUES ( ?)", ( dbVersion, ) )

def initDb( key, dbFolder ) :

    # create dbFolder if needed
    os.makedirs( dbFolder, exist_ok = True )

    # Create connection
    conn = sqlite3.connect( dbFolder + "/" + key + ".db" );

    cursor = conn.cursor()

    # versions initialized ?
    try :

        cursor.execute( "select * from versions" )

        # upgrade DB if needed
        upgradeDb( cursor )

    except sqlite3.OperationalError as e :
        # init tables
        if ( str( e ) == "no such table: versions" ) :
            initVersionTable( cursor )
            # Init tables
            initTables( cursor )
            # init base data
            initBaseData( conn )
            # Save (commit) the changes
            conn.commit()

    finally :
        cursor.close();

    # commit
    conn.commit();

    return conn;

def upgradeDb( cursor ):
    # Get current version
    curDbVersion = getDbVersion( cursor )
    while ( curDbVersion < DB_VERSION ) :

        curDbVersion += 1

        # execute patch
        if ( curDbVersion == 1 ) :
            initTables( cursor )
#         elif ( curDbVersion == 2 ) :
#             modifRunAddConfInfo( c )
#         elif ( curDbVersion == 3 ) :
#             modifAddDatasets( c )

        setDbVersion( cursor, curDbVersion )


def initVersionTable( cursor ):
    "Init version table"

    # create table dbVersion
    cursor.execute( '''CREATE TABLE IF NOT EXISTS versions
        (
           id integer PRIMARY KEY AUTOINCREMENT,
           version integer not null unique
         )'''
    )

    cursor.execute( '''CREATE INDEX IF NOT EXISTS idx_versions_id on versions( id ) ''' )
    cursor.execute( '''CREATE INDEX IF NOT EXISTS idx_versions_version on versions( version ) ''' )

    ## Create default value
    cursor.execute(
        "insert into versions values( null, ? )",
        (
            DB_VERSION,   # Initial version
        )
    )

def initTables( cursor ) :

    # Create table : current selection
    cursor.execute( '''CREATE TABLE IF NOT EXISTS selections
        (
           id integer PRIMARY KEY AUTOINCREMENT,
           jsonSelection text not null
         )'''
    )
    # Create default value
    cursor.execute( "insert into selections values( null, '{}' )" )

    # Create table - machines
    cursor.execute( '''CREATE TABLE IF NOT EXISTS machines
        (
           id integer PRIMARY KEY AUTOINCREMENT,
           name text not null unique
        )'''
    )

    # Create table - datasets
    cursor.execute( '''CREATE TABLE IF NOT EXISTS datasets
        (
           id integer PRIMARY KEY AUTOINCREMENT,
           name text not null unique,
           displayOrder integer not null,
           inMemory integer not null,
           pathHome text not null,
           pathTrn text not null,
           pathDev text not null
         )'''
    )

    # Create table - hyperparams
    cursor.execute( '''CREATE TABLE IF NOT EXISTS hyperparams
        (
           id integer PRIMARY KEY AUTOINCREMENT,
           json_hyper_params text not null unique
         )'''
    )

    # Create table - configs
    cursor.execute( '''CREATE TABLE IF NOT EXISTS configs
        (
           id integer PRIMARY KEY AUTOINCREMENT,
           name text,
           structure text,
           imageSize integer,
           idMachine not null,
           CONSTRAINT cs_unique0 UNIQUE (name, structure)
           FOREIGN KEY (idMachine) REFERENCES machines( id )
         )'''
    )

    # Create table - hyper-params and run selector
    cursor.execute( '''CREATE TABLE IF NOT EXISTS hpRunSelector
        (
           id integer not null PRIMARY KEY AUTOINCREMENT,
           idDataset integer not null REFERENCES datasets( id ),
           idConfig integer not null REFERENCES configs( id ),
           idHp integer not null REFERENCES hyperparams( id ),
           idRun integer REFERENCES runs( id )
        )'''
    )

    # Create table - run
    cursor.execute( '''CREATE TABLE IF NOT EXISTS runs
        (
           id integer PRIMARY KEY AUTOINCREMENT,
           idDataset integer,
           idConf integer,
           idHyperParams not null,
           date datetime DEFAULT CURRENT_TIMESTAMP,
           comment text,
           perf_index number,
           elapsed_second integer,
           train_accuracy number,
           dev_accuracy number,
           json_system_info text,
           json_data_info text,
           json_perf_info text,
           json_result_info text,
           json_conf_saved text,
           FOREIGN KEY (idDataset) REFERENCES datasets( id ),
           FOREIGN KEY (idConf) REFERENCES confs( id ),
           FOREIGN KEY (idHyperParams) REFERENCES hyperparams( id )
         )'''
    )

    # Indexes
    cursor.execute( '''CREATE INDEX IF NOT EXISTS idx_machines_id on machines( id ) ''' )
    cursor.execute( '''CREATE INDEX IF NOT EXISTS idx_machines_name on machines( name ) ''' )

    cursor.execute( '''CREATE INDEX IF NOT EXISTS idx_datasets_id on datasets( id ) ''' )
    cursor.execute( '''CREATE INDEX IF NOT EXISTS idx_datasets_displayOrder on datasets( displayOrder ) ''' )
    cursor.execute( '''CREATE INDEX IF NOT EXISTS idx_datasets_name on datasets( name ) ''' )

    cursor.execute( '''CREATE INDEX IF NOT EXISTS idx_configs_id on configs( id ) ''' )
    cursor.execute( '''CREATE INDEX IF NOT EXISTS idx_configs_structure on configs( structure ) ''' )

    cursor.execute( '''CREATE INDEX IF NOT EXISTS idx_hpRunSelector_id on hpRunSelector( id ) ''' )
    cursor.execute( '''CREATE UNIQUE INDEX IF NOT EXISTS idx_hpRunSelector_select on hpRunSelector( idDataset, idConfig )''' )

    cursor.execute( '''CREATE INDEX IF NOT EXISTS idx_hyperparams_id on hyperparams( id ) ''' )
    cursor.execute( '''CREATE INDEX IF NOT EXISTS idx_hyperparams_structure on hyperparams( id ) ''' )

    cursor.execute( '''CREATE INDEX IF NOT EXISTS idx_runs_id on runs( id ) ''' )
    cursor.execute( '''CREATE INDEX IF NOT EXISTS idx_runs_idDatasets on runs( idDataset ) ''' )
    cursor.execute( '''CREATE INDEX IF NOT EXISTS idx_runs_perf_index on runs( perf_index ) ''' )
    cursor.execute( '''CREATE INDEX IF NOT EXISTS idx_runs_elapsed_second on runs( elapsed_second ) ''' )
    cursor.execute( '''CREATE INDEX IF NOT EXISTS idx_runs_train_accuracy on runs( train_accuracy ) ''' )
    cursor.execute( '''CREATE INDEX IF NOT EXISTS idx_runs_dev_accuracy on runs( dev_accuracy ) ''' )

def initBaseData( conn ) :

    # Create "None machine
    addUniqueMachineName( conn, "None" )

def test( conn ):

     run = getRun( conn, 1 )
     print( "After :", str( run ) )

#     # Create config
#     hyperParams = { "beta": 100 }
#
#     idConfig = getOrCreateConfig( conn, "Hello conf 100", "[1]", hyperParams )
#
#     systemInfo = { "host": "12345678" }
#     dataInfo = { "data": "chats" }
#     perfInfo = { "perf": 4567 }
#
#
#     runHyperParams = { const.KEY_BETA: 10, const.KEY_KEEP_PROB: 0.5 }
#     idRun = createRun( conn, idConfig, runHyperParams )
#
#     updateRunBefore(
#         conn, idRun,
#         comment="comment",
#         system_info=systemInfo, data_info=dataInfo
#     )
#
#     run = getRun( conn, idRun )
#     print( "Before:", str( run ) )
#
#     updateRunAfter(
#         conn, idRun,
#         perf_info = perfInfo, result_info={ "errors": [1,2,3] },
#         perf_index=10, elapsed_second=20, train_accuracy=0.5, dev_accuracy=0.9
#     )
#
#     run = getRun( conn, idRun )
#     print( "After :", str( run ) )
