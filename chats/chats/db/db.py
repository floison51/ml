'''
Created on 28 avr. 2018
Machine Learning DB
@author: fran
'''
import sqlite3
import os
import datetime
import json

def createConfig( conn, name, structure, hyper_params ) :

    # get hyparams
    idHyperParams = getOrCreateHyperParams( conn, hyper_params )

    c = conn.cursor();

    cursor = c.execute(
        "INSERT INTO configs VALUES ( null, ?, ?, ? )",
        ( name, structure, idHyperParams, )
    )

    id = cursor.lastrowid

    c.close();

    # commit
    conn.commit()

    return id

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
        result[ "hyper_params" ]  = hyper_params

    c.close();

    return result

def getOrCreateHyperParams( conn, hyper_params ) :

    c = conn.cursor();

    # JSon conversion
    json_hyper_params   = json.dumps( hyper_params )

    # get existing, if anay
    cursor = c.execute(
        "select id from hyperparams where json_hyper_params=?",
        ( json_hyper_params, )
    )

    # For some reason, cursor.rowcount is NOK, use another method
    idResult = -1
    for ( id ) in cursor:
        idResult = id[ 0 ]

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

    c = conn.cursor()
    
    cursor = c.execute(
        "select c.* from configs c where c.id=? ",
        ( idConfig, )
    )

    result = {}

    for row in cursor :
        result[ "id" ]               = row[ 0 ]
        result[ "name" ]             = row[ 1 ]
        result[ "structure" ]        = row[ 2 ]
        result[ "idHyperParams" ]    = row[ 3 ]

    c.close();

    return result

def getOrCreateConfig( c, name, structure, hyperParams ) :

    c = conn.cursor();

    # get existing, if anay
    cursor = c.execute(
        "select id from configs where name=? and structure=?" ,
        ( name, structure, )
    )

    # For some reason, cursor.rowcount is NOK, use another method
    idResult = -1
    for ( id ) in cursor:
        idResult = id[ 0 ]

    if ( idResult < 0 ) :
        # None, create
        idResult = createConfig( conn, name, structure, hyperParams )

    c.close()

    return idResult;

def getConfigsWithMaxDevAccuracy( conn ) :

    c = conn.cursor()
    
    # Update run
    cursor = c.execute(
        "select c.id, c.name, c.structure, ( select max(r.dev_accuracy) from runs r where r.idConf=c.id ) from configs c " +
        "order by c.id asc"
    )

    results = []

    for row in cursor :
        
        result = []
        
        result.append( row[ 0 ] )
        result.append( row[ 1 ] )
        result.append( row[ 2 ] )
        result.append( row[ 3 ] )

        results.append( result )
        
    c.close();

    return results


def createRun( conn, idConfig ) :

    c = conn.cursor();

    # get config
    config = getConfig( conn, idConfig )

    cursor = c.execute( '''
        INSERT INTO runs ( idConf, idHyperParams, date ) VALUES ( ?, ?, ? )''',
        ( config[ "id" ], config[ "idHyperParams" ], datetime.datetime.now(), )
    )

    id = cursor.lastrowid

    c.close();

    return id

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

def getRun( conn, idRun ) :

    c = conn.cursor();

    # Update run
    cursor = c.execute( '''
        select * from runs where id=?''',
        (idRun,)
    )

    result = {}

    for row in cursor :
        result[ "id" ]              = row[ 0 ]
        result[ "idConf" ]          = row[ 1 ]
        result[ "idHyperParams" ]   = row[ 2 ]
        result[ "dateTime" ]        = row[ 3 ]
        result[ "comment" ]         = row[ 4 ]
        result[ "perf_index" ]      = row[ 5 ]
        result[ "elapsed_second" ]  = row[ 6 ]
        result[ "train_accuracy" ]  = row[ 7 ]
        result[ "dev_accuracy" ]    = row[ 8 ]
        result[ "system_info" ]     = json.loads( row[ 9 ] )
        result[ "data_info" ]       = json.loads( row[ 10 ] )
        result[ "perf_info" ]       = json.loads( row[ 11 ] )
        result[ "result_info" ]     = json.loads( row[ 12 ] )

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
           idHyperParams not null,
           CONSTRAINT cs_unique0 UNIQUE (name, structure)
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
    c.execute( '''CREATE INDEX idx_configs_id on configs( id ) ''' )
    c.execute( '''CREATE INDEX idx_configs_structure on configs( structure ) ''' )

    c.execute( '''CREATE INDEX idx_hyperparams_id on hyperparams( id ) ''' )
    c.execute( '''CREATE INDEX idx_hyperparams_structure on hyperparams( id ) ''' )

    c.execute( '''CREATE INDEX idx_runs_id on runs( id ) ''' )
    c.execute( '''CREATE INDEX idx_runs_perf_index on runs( perf_index ) ''' )
    c.execute( '''CREATE INDEX idx_runs_elapsed_second on runs( elapsed_second ) ''' )
    c.execute( '''CREATE INDEX idx_runs_train_accuracy on runs( train_accuracy ) ''' )
    c.execute( '''CREATE INDEX idx_runs_dev_accuracy on runs( dev_accuracy ) ''' )

if __name__ == '__main__':

    DB_DIR = "C:/Users/frup82455/git/ml/chats/chats/run/db/chats"
    APP_KEY = "chats"

        # Init DB
    print( "Db dir:", DB_DIR )
    with initDb( APP_KEY, DB_DIR ) as conn :

        # Create config
        hyperParams = { "beta": 0 }

        idConfig = getOrCreateConfig( conn, "Hello conf6", "[2]", hyperParams )

        systemInfo = { "host": "12345678" }
        dataInfo = { "data": "chats" }
        perfInfo = { "perf": 4567 }

        idRun = createRun( conn, idConfig )

        updateRunBefore(
            conn, idRun,
            comment="comment",
            system_info=systemInfo, data_info=dataInfo
        )

        run = getRun( conn, idRun )
        print( "Before:", str( run ) )

        updateRunAfter(
            conn, idRun,
            perf_info = perfInfo, result_info={ "errors": [1,2,3] },
            perf_index=10, elapsed_second=20, train_accuracy=0.5, dev_accuracy=0.1
        )

        run = getRun( conn, idRun )
        print( "After :", str( run ) )
