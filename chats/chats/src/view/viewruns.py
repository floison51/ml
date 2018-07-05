'''
Created on 28 avr. 2018
Machine Learning runs
@author: fran
'''
from tkinter import *
from tkinter.ttk import Treeview

import view.view as view
import const.constants as const
import db.db as db

class ViewRunsWindow ( view.MyDialog ) :

    def __init__( self, boss, callbackFct, **options ) :
        view.MyDialog.__init__( self, boss, callbackFct, **options )

    def run( self, idDataset, idConfig, datasetName, runs ) :
        "View or Update hyper parameters, show best hyper params"

        # record idConf and runs
        self.idDataset = idDataset
        self.idConfig = idConfig
        self.runs = runs

        frameTop = Frame( self, relief=GROOVE )
        frameTop.pack(side=TOP, padx=30, pady=30)

        label = Label( frameTop, text="Runs for dataset '" + datasetName + "', config '" + str( idConfig ) + "'" );
        label.pack()

        frameForm = Frame( self, relief=GROOVE )
        frameForm.pack( padx=30, pady=30)

        self.tree = Treeview( frameForm, columns = const.RunsDico.DISPLAY_FIELDS )

        # Add col names excepted for info fields shows as lines
        #Skip id col
        for colName in const.RunsDico.DISPLAY_FIELDS :

            if ( colName.endswith( "_info" ) ) :
                #skip
                continue

            self.tree.heading( colName, text=colName )
            self.tree.column( colName, stretch=YES )

        self.tree.grid( row=len( const.RunsDico.DISPLAY_FIELDS ), sticky='nsew' )
        self.treeview = self.tree

        # Add data
        self.addData()

    def addData( self ) :

        iLine = 1

        for run in self.runs :

            strIdRun = str( run["id" ] )

            cols = []
            for colName in const.RunsDico.DISPLAY_FIELDS :

                # Add col values excepted for info fields shows as lines
                if ( colName.endswith( "_info" ) ) :
                    #skip
                    continue

                col = "<None>"
                if ( colName in run ) :
                    col = run[ colName ]
                    #format?
                    strFormat = const.RunsDico.CARAC[ colName ][2 ]
                    if ( strFormat != None ) :
                        col = strFormat.format( col )
                    else :
                        col = str( run[ colName ] )                        

                cols.append( col )

            idLine = self.treeview.insert( '', iLine, text="Run " + strIdRun, values=cols )
            iLine += 1

            # Add lines for json data
            for colName in const.RunsDico.DISPLAY_FIELDS :

                # Add col values excepted for info fields shows as lines
                if ( colName.endswith( "_info" ) ) :

                    col = "<None>"
                    if ( colName in run ) :
                        col = str( run[ colName ] )

                    self.treeview.insert( idLine, "end", colName + strIdRun, text=colName, values=( col ) )
