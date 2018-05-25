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

    def run( self, idConfig, runs ) :
        "View or Update hyper parameters, show best hyper params"

        # record idConf and runs
        self.idConfig = idConfig
        self.runs = runs
        
        frameTop = Frame( self, relief=GROOVE )
        frameTop.pack(side=TOP, padx=30, pady=30)

        label = Label( frameTop, text="Runs for config " + str( idConfig ) );
        label.pack()

        frameForm = Frame( self, relief=GROOVE )
        frameForm.pack( padx=30, pady=30)

        self.tree = Treeview( frameForm, columns = const.RunsDico.DISPLAY_FIELDS )
        
        for colName in const.RunsDico.DISPLAY_FIELDS :
            self.tree.heading( colName, text=colName )
            self.tree.column( colName, stretch=YES )

        self.tree.grid( row=4, columnspan=len( const.RunsDico.DISPLAY_FIELDS ), sticky='nsew' )
        self.treeview = self.tree
        
        # Add data
        self.addData()
        
    def addData( self ) :
        
        for run in self.runs :
            
            for colName in const.RunsDico.DISPLAY_FIELDS :
                col = run[ colName ]
                self.treeview.insert( '', 'end', text="Item_1", values=( "10 mg", "100" ) )
