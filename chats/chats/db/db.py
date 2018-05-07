'''
Created on 28 avr. 2018
Machine Learning DB
@author: fran
'''
import sqlite3
import os

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

    # close
    c.close()
    conn.close()
    
def initTables( c ) :
    # Create table
    c.execute( '''CREATE TABLE runs
        (
          id integer PRIMARY KEY, 
          date datetime not null, 
          system_info text not null,
          hyper_params text not null,
          data_info text not null,
          perf_info text not null,
          duration_s integer not null,
          train_accuracy number not null,
          dev_accuracy number not null
        )'''
    )
    
    # Indexes
    c.execute( '''CREATE INDEX idx_runs_train_accuracy on runs( train_accuracy ) ''' )
    c.execute( '''CREATE INDEX idx_runs_dev_accuracy on runs( dev_accuracy ) ''' )
