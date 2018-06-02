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

def createConfig( conn, name, structure, imageSize, machineName, hyper_params ) :

    # Id machine
    idMachine = getIdMachineByName( conn, machineName )

    # get hyparams
    idHyperParams = getOrCreateHyperParams( conn, hyper_params )

    c = conn.cursor();

    cursor = c.execute(
        "INSERT INTO configs VALUES ( null, ?, ?, ?, ?, ? )",
        ( name, structure, imageSize, idMachine, idHyperParams, )
    )

    idConfig = cursor.lastrowid

    c.close();

    return idConfig

def getHyperParams( conn, idHyperParams ) :

    c = conn.cursor();

    # get existing, if anay
    cursor = c.execute(
        "select * from hyperparams where id=?",
        ( idHyperParams, )
    )

    result = {}

    for row in cursor :
        result[ "id" ]    = row[ 0 ]
        json_hyper_params = row[ 1 ]

        hyper_params = json.loads( json_hyper_params )
        result[ const.KEY_DICO_HYPER_PARAMS ]  = hyper_params

    c.close();

    return result

def getBestHyperParams( conn, idConfig ) :

    c = conn.cursor();

    # get existing, if anay
    cursor = c.execute(
        "select r.idHyperParams, max(r.dev_accuracy) from configs c, runs r where ( c.id=? and r.idConf=c.id )",
        ( idConfig, )
    )

    devAccuracy = None
    hyperParams = {}

    idHyperParams = None
    for row in cursor :
        idHyperParams = row[ 0 ]
        devAccuracy   = row[ 1 ]

    c.close();

    if ( idHyperParams != None ) :
        hyperParams = getHyperParams( conn, idHyperParams )

    return ( hyperParams, devAccuracy )

def getOrCreateHyperParams( conn, hyper_params ) :

    c = conn.cursor();

    # Sort hyperparams by key
    sorted_hyper_params = OrderedDict( sorted( hyper_params.items(), key=lambda t: t[0] ) )
    # JSon conversion
    json_hyper_params   = json.dumps( sorted_hyper_params )

    # get existing, if anay
    cursor = c.execute(
        "select id from hyperparams where json_hyper_params=?",
        ( json_hyper_params, )
    )

    # For some reason, cursor.rowcount is NOK, use another method
    idResult = -1
    for ( hp ) in cursor:
        idResult = hp[ 0 ]

    if ( idResult < 0 ) :
        # None, create
        cursor = c.execute(
            "INSERT INTO hyperparams VALUES ( null, ? )",
            ( json_hyper_params, )
        )

        idResult = cursor.lastrowid

    c.close()

    return idResult;

def getConfig( conn, idConfig ) :

    from model.mlconfig import MlConfig
    
    c = conn.cursor()

    cursor = c.execute(
        "select c.* from configs c where c.id=? ",
        ( idConfig, )
    )

    result = MlConfig()

    for row in cursor :
        
        iCol = 0;
        
        for colName in const.ConfigsDico.OBJECT_FIELDS :    
            result[ colName ] = row[ iCol ]
            iCol += 1

    c.close();

    return result

def getOrCreateConfig( conn, name, structure, hyperParams ) :

    c = conn.cursor();

    # get existing, if anay
    cursor = c.execute(
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

    c.close()

    return idResult;

def updateConfig( conn, config ) :

    c = conn.cursor();

    # Update config
    updateStatement = \
        "update configs set " + \
             "name=?, " + \
             "structure=?, " + \
             "imageSize=?, " + \
             "idMachine=?, " + \
             "idHyperParams=? " + \
             "where id=?"

    c.execute(
        updateStatement,
        (
            config[ "name" ], config[ "structure" ], config[ "imageSize" ], config[ "idMachine" ], config[ "idHyperParams" ], config[ "id" ]
        )
    )

    c.close();

def deleteConfig( conn, idConf ) :

    c = conn.cursor();

    # Update config
    deleteStatement = "delete from configs where id=?"
    c.execute(
        deleteStatement,
        ( idConf, )
    )

    c.close();

def getConfigsWithMaxDevAccuracy( conn, idConfig = None ) :

    from model.mlconfig import MlConfig

    c = conn.cursor()
    parameters = ()

    statement = \
        "select c.id as id, c.name as name, " + \
        "( select m.name from machines m where m.id=c.idMachine ) as machine, " + \
        "c.imageSize as imageSize, c.structure as structure, " + \
        "( select max(r.dev_accuracy) " + \
        "from runs r where r.idConf=c.id ) as bestAccuracy from configs c"
    if ( idConfig != None ) :
        statement += " where c.id=?"
        parameters = ( idConfig, )

    statement += " order by c.id asc"

    # Update run
    cursor = c.execute(
        statement,
        parameters
    )

    results = []

    for row in cursor :

        result = MlConfig()

        iCol = 0
        
        for colName in const.ConfigsDico.DISPLAY_FIELDS : 
            result[ colName ] = row[ iCol ]
            iCol += 1

        # add result
        results.append( result )

    c.close();

    return results

def getMachineNames( conn ) :

    c = conn.cursor();

    # Update run
    cursor = c.execute(
        "select name from machines"
    )

    result = []

    for row in cursor :
        result.append( row[ 0 ] )

    c.close();

    return result

def getIdMachineByName( conn, machineName ):

    c = conn.cursor();

    # Update run
    cursor = c.execute(
        "select id from machines where name=?",
        ( machineName, )
    )

    result = None

    for row in cursor :
        result = row[ 0 ]

    c.close();

    return result

def getMachineNameById( conn, idMachine ):

    c = conn.cursor();

    # Update run
    cursor = c.execute(
        "select name from machines where id=?",
        ( idMachine, )
    )

    result = None

    for row in cursor :
        result = row[ 0 ]

    c.close();

    return result

def getMachineIdByName( conn, machineName ):

    c = conn.cursor();

    # Update run
    cursor = c.execute(
        "select id from machines where name=?",
        ( machineName, )
    )

    result = None

    for row in cursor :
        result = row[ 0 ]

    c.close();

    return result

def addUniqueMachineName( conn, machineName ) :

    c = conn.cursor();
    
    # exists?
    idMachine = getMachineIdByName( conn, machineName )
    
    if ( idMachine == None ) :
        cursor = c.execute( '''
            INSERT INTO machines ( name ) VALUES ( ? )''',
            ( machineName, )
        )
    
        idMachine = cursor.lastrowid

    c.close();

    return idMachine



def createRun( conn, idConfig, runHyperParams ) :

    c = conn.cursor();

    # get config
    config = getConfig( conn, idConfig )

    # Get hyperparams
    idRunHyperParams = getOrCreateHyperParams( conn, runHyperParams );

    cursor = c.execute( '''
        INSERT INTO runs ( idConf, idHyperParams, date ) VALUES ( ?, ?, ? )''',
        ( config[ "id" ], idRunHyperParams, datetime.datetime.now(), )
    )

    idRun = cursor.lastrowid

    c.close();

    return idRun

def updateRun( conn, idRun, runHyperParams ) :

    c = conn.cursor();

    # Get hyperparams
    idRunHyperParams = getOrCreateHyperParams( conn, runHyperParams );

    cursor = c.execute( \
        "update runs set idHyperParams=? where id=?", \
        ( idRunHyperParams, idRun, )
    )

    idRun = cursor.lastrowid

    c.close();

    return idRun

def updateRunBefore(
    conn, idRun,
    comment = "?",
    system_info = {}, data_info = {},
) :

    c = conn.cursor();

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

    c.execute(
        updateStatement,
        (
            comment,
            json_system_info, json_data_info, json_perf_info, json_result_info,
            -1, -1, -1,
            idRun,
        )
    )

    c.close();

def updateRunAfter(
    conn, idRun,
    perf_info = {}, result_info={},
    perf_index = -1, elapsed_second =- 1, train_accuracy = -1, dev_accuracy = -1
) :

    c = conn.cursor();

    # JSon conversion
    json_perf_info      = json.dumps( perf_info )
    json_result_info    = json.dumps( result_info )

    # Update run
    updateStatement = \
        "update runs set " + \
            "json_perf_info=?, json_result_info=?," + \
             "perf_index=?, elapsed_second=?, train_accuracy=?, dev_accuracy=? " + \
             "where id=?"

    c.execute(
        updateStatement,
        (
            json_perf_info, json_result_info,
            perf_index, elapsed_second, train_accuracy, dev_accuracy,
            idRun,
        )
    )

    c.close();

    # Save (commit) the changes
    conn.commit()


def getRunFromRow(row):
    # TODO : use dico
    result = {}
    result["id"] = row[0]
    result["idConf"] = row[1]
    result["idHyperParams"] = row[2]
    result["dateTime"] = row[3]
    result["comment"] = row[4]
    result["perf_index"] = row[5]
    result["elapsed_second"] = row[6]
    result["train_accuracy"] = row[7]
    result["dev_accuracy"] = row[8]
    result["system_info"] = json.loads(row[9])
    result["data_info"] = json.loads(row[10])
    result["perf_info"] = json.loads(row[11])
    result["result_info"] = json.loads(row[12])
    return result

def getRuns( conn, idConf ) :

    c = conn.cursor();

    # Update run
    cursor = c.execute( '''
        select * from runs where idConf=?''',
        (idConf,)
    )

    results = []

    # TODO use dico
    for row in cursor :
        
        result = getRunFromRow( row )

        results.append( result )
        
    c.close();

    return results

def getRun( conn, idRun ) :

    c = conn.cursor();

    # Update run
    cursor = c.execute( '''
        select * from runs where id=?''',
        (idRun,)
    )

    result = None

    for row in cursor :
        result = getRunFromRow( row )

    c.close();

    return result

def initDb( key, dbFolder ) :

    # create dbFolder if needed
    os.makedirs( dbFolder, exist_ok = True )

    # Create connection
    conn = sqlite3.connect( dbFolder + "/" + key + ".db" );

    c = conn.cursor();

    # initialized?
    try :
        c.execute( "select * from runs" )
    except sqlite3.OperationalError as e :
        # init tables
        if ( str( e ) == "no such table: runs" ) :
            initTables( c )
            # Save (commit) the changes
            conn.commit()

    c.close();

    return conn;

def initTables( c ) :

    # Create table - machines
    c.execute( '''CREATE TABLE IF NOT EXISTS machines
        (
           id integer PRIMARY KEY AUTOINCREMENT,
           name text not null unique
         )'''
    )

    # Create table - hyperparams
    c.execute( '''CREATE TABLE IF NOT EXISTS hyperparams
        (
           id integer PRIMARY KEY AUTOINCREMENT,
           json_hyper_params text not null unique
         )'''
    )

    # Create table - configs
    c.execute( '''CREATE TABLE IF NOT EXISTS configs
        (
           id integer PRIMARY KEY AUTOINCREMENT,
           name text,
           structure text,
           imageSize integer,
           idMachine not null,
           idHyperParams not null,
           CONSTRAINT cs_unique0 UNIQUE (name, structure)
           FOREIGN KEY (idMachine) REFERENCES machines( id )
           FOREIGN KEY (idHyperParams) REFERENCES hyperparams( id )
         )'''
    )

    # Create table - run
    c.execute( '''CREATE TABLE IF NOT EXISTS runs
        (
           id integer PRIMARY KEY AUTOINCREMENT,
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
           FOREIGN KEY (idConf) REFERENCES confs( id ),
           FOREIGN KEY (idHyperParams) REFERENCES hyperparams( id )
         )'''
    )

    # Indexes
    c.execute( '''CREATE INDEX idx_machines_id on machines( id ) ''' )
    c.execute( '''CREATE INDEX idx_machines_name on machines( name ) ''' )

    c.execute( '''CREATE INDEX idx_configs_id on configs( id ) ''' )
    c.execute( '''CREATE INDEX idx_configs_structure on configs( structure ) ''' )

    c.execute( '''CREATE INDEX idx_hyperparams_id on hyperparams( id ) ''' )
    c.execute( '''CREATE INDEX idx_hyperparams_structure on hyperparams( id ) ''' )

    c.execute( '''CREATE INDEX idx_runs_id on runs( id ) ''' )
    c.execute( '''CREATE INDEX idx_runs_perf_index on runs( perf_index ) ''' )
    c.execute( '''CREATE INDEX idx_runs_elapsed_second on runs( elapsed_second ) ''' )
    c.execute( '''CREATE INDEX idx_runs_train_accuracy on runs( train_accuracy ) ''' )
    c.execute( '''CREATE INDEX idx_runs_dev_accuracy on runs( dev_accuracy ) ''' )

    ## Create default values
    c.execute(
        "insert into machines values( null, ? )",
        ( 
            const.MachinesDico.CARAC[ "name"  ][ 1 ],   #default value for name
        ) 
    )

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
