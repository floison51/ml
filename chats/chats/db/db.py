'''
Created on 28 avr. 2018
Machine Learning DB
@author: fran
'''
import sqlite3
import os
import datetime
import json

def createRun( conn ) :
    
    c = conn.cursor();
    
    cursor = c.execute( '''
        INSERT INTO runs VALUES ( null, ?, null, null, null, null, null, null, null, null, null, null )''',
        ( datetime.datetime.now(), )
    )
    
    id = cursor.lastrowid
    
    c.close();
    
    # Save (commit) the changes
    conn.commit()
    
    return id
    
def updateRunBefore( 
    conn, idRun, 
    structure = "?", comment = "?", 
    system_info = {}, hyper_params = {}, data_info = {},
) :

    c = conn.cursor();
    
    # JSon conversion
    json_system_info    = json.dumps( system_info )
    json_hyper_params   = json.dumps( hyper_params )
    json_data_info      = json.dumps( data_info )
    json_perf_info      = json.dumps( {} )
    json_result_info    = json.dumps( {} )
    
    # Update run
    updateStatement = \
        "update runs set " + \
             "structure=?, comment=?, " + \
             "json_system_info=?, json_hyper_params=?, json_data_info=?, json_perf_info=?, json_result_info=?, " + \
             "perf_index=?, train_accuracy=?, dev_accuracy=? " + \
             "where id=?"

    c.execute( 
        updateStatement,
        ( 
            str( structure ), comment,
            json_system_info, json_hyper_params, json_data_info, json_perf_info, json_result_info,
            -1, -1, -1,
            idRun, 
        )
    )
    
    c.close();
    
    # Save (commit) the changes
    conn.commit()
        
def updateRunAfter( 
    conn, idRun, 
    perf_info = {}, result_info={}, 
    perf_index = -1, train_accuracy = -1, dev_accuracy = -1
) :

    c = conn.cursor();
    
    # JSon conversion
    json_perf_info      = json.dumps( perf_info )
    json_result_info    = json.dumps( result_info )
    
    # Update run
    updateStatement = \
        "update runs set " + \
            "json_perf_info=?, json_result_info=?," + \
             "perf_index=?, train_accuracy=?, dev_accuracy=? " + \
             "where id=?"

    c.execute( 
        updateStatement,
        ( 
            json_perf_info, json_result_info,
            perf_index, train_accuracy, dev_accuracy,
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
        result[ "dateTime" ]        = row[ 1 ]
        result[ "structure" ]       = row[ 2 ]
        result[ "comment" ]         = row[ 3 ]
        result[ "perf_index" ]      = row[ 4 ]
        result[ "train_accuracy" ]  = row[ 5 ]
        result[ "dev_accuracy" ]    = row[ 6 ]
        result[ "system_info" ]     = json.loads( row[ 7 ] )
        result[ "hyper_params" ]    = json.loads( row[ 8 ] )
        result[ "data_info" ]       = json.loads( row[ 9 ] )
        result[ "perf_info" ]       = json.loads( row[ 10 ] )
        result[ "result_info" ]     = json.loads( row[ 11 ] )
        
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
    
    # Create table
    c.execute( '''CREATE TABLE runs
        (
           id integer PRIMARY KEY AUTOINCREMENT, 
           date datetime DEFAULT CURRENT_TIMESTAMP,
           structure text,
           comment text,
           perf_index number,
           train_accuracy number,
           dev_accuracy number,
           json_system_info text,
           json_hyper_params text,
           json_data_info text,
           json_perf_info text,
           json_result_info text
         )'''
    )
    
    # Indexes
    c.execute( '''CREATE INDEX idx_runs_id on runs( id ) ''' )
    c.execute( '''CREATE INDEX idx_runs_perf_index on runs( perf_index ) ''' )
    c.execute( '''CREATE INDEX idx_runs_train_accuracy on runs( train_accuracy ) ''' )
    c.execute( '''CREATE INDEX idx_runs_dev_accuracy on runs( dev_accuracy ) ''' )
    
if __name__ == '__main__':

    DB_DIR = "C:/Users/frup82455/git/ml/chats/chats/run/db/chats"
    APP_KEY = "chats"
    
        # Init DB
    print( "Db dir:", DB_DIR )
    with initDb( APP_KEY, DB_DIR ) as conn : 

        # Create run
        idRun = createRun( conn )
        
        systemInfo = { "host": "12345678" }
        hyperParams = { "toto": "titi" }
        dataInfo = { "data": "chats" }
        perfInfo = { "perf": 4567 }
        
        updateRunBefore( 
            conn, idRun, 
            structure="[2]", comment="comment",
            system_info=systemInfo, hyper_params=hyperParams, data_info=dataInfo
        )
        
        run = getRun( conn, idRun )
        print( "Before:", str( run ) )
         
        updateRunAfter( 
            conn, idRun, 
            perf_info = perfInfo, result_info={ "errors": [1,2,3] },
            perf_index=10, train_accuracy=0.5, dev_accuracy=0.75
        )
        
        run = getRun( conn, idRun )    
        print( "After :", str( run ) )
    