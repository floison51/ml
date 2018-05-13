'''
Created on 28 avr. 2018
Machine Learning configuration
@author: fran
'''
from tkinter import *
import const.constants as const
from tables.table import Cols

buttonsToUpdate = []

class MainWindow ( Tk ):
    
    def __init__( self ) :
        
        Tk.__init__(self)

        # From conf radio-buttons
        self.confSelected = StringVar()
        self.confSelected.set( "None" )
        
    def show( self, configs, configDoer, hpDoer ):
    
        frameTop = Frame( self, relief=GROOVE )
        frameTop.pack(side=TOP, padx=30, pady=30)
    
        label = Label( frameTop, text="Cat's Recognition Machine Learning" )
        label.pack()
    
        frameConfigs = LabelFrame( self, text="Configurations", padx=20, pady=20)
        self.buildConfigGrid( frameConfigs, configs, configDoer, hpDoer )
        frameConfigs.pack(fill="both", expand="yes", padx=10, pady=10)
    
        frameButtons = Frame(self, borderwidth=0 )
        frameButtons.pack( side=BOTTOM, padx=10, pady=10, fill='both', expand=True )
    
        buttonRun = Button( frameButtons, text="Run", command=self.quit, state=DISABLED )
        buttonRun.pack( side="left", padx=40 )
    
        buttonsToUpdate.append( buttonRun )
    
        buttonCancel=Button( frameButtons, text="Cancel", command=self.quit)
        buttonCancel.pack( side="right", padx=40  )
    
        self.mainloop()


    def addRowConfig(self, hpDoer, frameConfigsTable, iCol, label, Label, W, iRow, idConf, config):
        cols = []
        idConf = config[0]
    # Radio button
        radioButton = Radiobutton(frameConfigsTable, variable=self.confSelected, value=idConf, command=self.confRbClicked)
        cols.append(radioButton)
        radioButton.grid(row=iRow, column=1)
        iCol = 2
        for item in config:
            label = Label(frameConfigsTable, text=item, borderwidth=1)
            cols.append(label)
            label.grid(row=iRow, column=iCol, sticky=W, padx=10)
            iCol += 1
        
    # Button to show hyper params
        buttonShowHP = Button(frameConfigsTable, text="Update", command=(lambda idConf=idConf:hpDoer.updateHyperParams(self, idConf)))
        cols.append(buttonShowHP)
        buttonShowHP.grid(row=iRow, column=iCol)
        self.rows[idConf] = cols
        return padx, borderwidth, text, command, row, column, idConf

    def buildConfigGrid( self, frameConfigs, configs, configDoer, hpDoer ):
    
    #     frameConfigsGlobalTable = LabelFrame( frameConfigs, padx=10, pady=10)
    #     frameConfigsGlobalTable.pack( side = "top" )
    #
    #     scrollbar = Scrollbar( frameConfigsGlobalTable )
    #     scrollbar.pack( side = RIGHT, fill = Y )
    
    #     frameConfigsTable = LabelFrame( frameConfigsGlobalTable )
    
        self.rows = {}
        
        frameConfigsTable = LabelFrame( frameConfigs, padx=10, pady=10)
    
        # show window
        labels = { 1: "", 2: "Id", 3: "Name", 4: "Structure", 5 : "Best DEV\nAccuracy", 6: "Hyper Params" }
    
        for iCol in range( 1, 7 ) :
            label = Label( frameConfigsTable, text=labels[ iCol ], borderwidth=1 ).grid( row=1, column=iCol, sticky=W, padx=10 )
    
        iRow = 2
        
        idConf = None
        
        for config in configs :
    
            padx, borderwidth, text, command, row, column, idConf = self.addRowConfig(hpDoer, frameConfigsTable, iCol, label, Label, W, iRow, idConf, config)
            
            iRow += 1
    
        frameConfigsButtons = LabelFrame( frameConfigs, padx=10, pady=10, borderwidth=0 )
        frameConfigsButtons.pack( side = "bottom", fill='both', expand=True )
    
        buttonNewConfig = Button( frameConfigsButtons, text="New", command=(lambda : ( configDoer.newConfig( self ) )  ) )
        buttonNewConfig.grid( row=1, column=1, padx=10 )
#         buttonUpdateConfig = Button( frameConfigsButtons, text="Update", command=configDoer.updateConfig( self ), state=DISABLED )
#         buttonUpdateConfig.grid( row=1, column=2, padx=10 )
        buttonDeleteConfig = Button( frameConfigsButtons, text="Delete", command=(lambda idConf=idConf : ( configDoer.deleteConfig( self, idConf ) ) ), state=DISABLED )
        buttonDeleteConfig.grid( row=1, column=3, padx=10 )
    
#         buttonsToUpdate.append( buttonUpdateConfig )
        buttonsToUpdate.append( buttonDeleteConfig )
        
        # Last one is a button
        #button = Button(frameConfigs, text=labels[ len( labels ) ], borderwidth=1).grid( row=1, column=len( labels ) )
        #button.pck()
        
        frameConfigsTable.pack( side = "top" )

    
    def deleteConfigGrid( self, idConfig ):
        cols = self.rows[ idConfig ]
        for col in cols :
            col.destroy()
        # delete line
        self.rows.remove( idConfig )
        
    def enableEntry( self, entry ):
        entry.configure( state=NORMAL )
        entry.update()
    
    def confRbClicked( self ):
        for button in buttonsToUpdate :
            self.enableEntry( button )
    
class MyDialog( Toplevel ):

    def __init__( self, boss, callbackFct, **options ) :
        Toplevel.__init__( self, boss, **options )
        
        # Modal window
        ## disable parent window
        boss.wm_attributes( "-disabled", True )
        self.transient( boss )
        
        self.callbackFct = callbackFct
        self.result = None
        self.inputs = {}
        
    def do_close( self ) :
        ## enable parent window
        self.master.wm_attributes( "-disabled", False )
        self.destroy()
        # update callback
        self.callbackFct( self, self.result )
        
    def buttonCancelClicked( self ) : 
        # Bye
        result = None
        self.do_close()
        
    def buttonOkClicked( self ) : 
        
        # Read values from form
        self.result = {}
        for key, value in self.inputs.items() :
            self.result[ key ] = value.get()

        # Bye
        self.do_close()
        
class ViewOrUpdateHyperParamsWindow ( MyDialog ) :
    
    def __init__( self, boss, callbackFct, **options ) :
        MyDialog.__init__( self, boss, callbackFct, **options )
        
    def run( self, dbHyperParams, dbBestHyperParams, bestDevAccuracy ) :
        "View or Update hyper parameters, show best hyper params"
    
        frameTop = Frame( self, relief=GROOVE )
        frameTop.pack(side=TOP, padx=30, pady=30)
    
        label = Label( frameTop, text="Hyper parameter form" )
        label.pack()
    
        frameForm = Frame( self, relief=GROOVE )
        frameForm.pack( padx=30, pady=30)
    
        frameButtons = LabelFrame( self, borderwidth=0 )
        frameButtons.pack( side=BOTTOM, padx=30, pady=30, fill='both', expand=True )
    
        #Get real hyper params
        hyperParams     = dbHyperParams[ const.KEY_DICO_HYPER_PARAMS ]
        bestHyperParams = dbBestHyperParams.get( const.KEY_DICO_HYPER_PARAMS, {} )
        
        # Add default values
        for ( key, hpCarac ) in const.HyperParamsDico.CARAC.items() :
            if not key in hyperParams :
                # add default value
                hyperParams[ key ] = hpCarac[ 1 ]
        
        # Add dico entries
        iRow = 1

        labelText1 = "Current\nhyper params"
        labelText2 = "Best\nhyper params\nDEV accuracy=" + str( bestDevAccuracy )
        Label( frameForm, text=labelText1, borderwidth=1 ).grid( row=iRow, column=1, sticky=W, padx=10 )
        Label( frameForm, text=labelText2, borderwidth=1 ).grid( row=iRow, column=3, sticky=W, padx=10 )
    
        # Table labels
        iRow += 1
        
        for key in hyperParams.keys() :
    
            # Label value
            # Label
            label = Label( frameForm, text=key, borderwidth=1 ).grid( row=iRow, column=1, sticky=W, padx=10 )
    
            # input : type depends on hp name
            inputVar = self.getInputVar( key )
            
            input = Entry( frameForm, textvariable=inputVar ).grid( row=iRow, column=2, sticky=W, padx=10 )
            self.inputs[ key ] = inputVar

            value = hyperParams.get( key )
            inputVar.set( str( value ) )
            
            # Best hyper param
            bestValue = bestHyperParams.get( key, "" )
            Label( frameForm, text=bestValue, borderwidth=1 ).grid( row=iRow, column=3, sticky=W, padx=10 )
            
            iRow += 1
    
        buttonSave   = Button( frameButtons, text="Save"  , command=self.buttonOkClicked    , state=NORMAL )
        buttonCancel = Button( frameButtons, text="Cancel", command=self.buttonCancelClicked, state=NORMAL )
        
        buttonSave.pack( side=LEFT, padx=40 )
        buttonCancel.pack( side=RIGHT, padx=40 )
        
    def getInputVar( self, hpName ) :
        # use dico
        hpCarac = const.HyperParamsDico.CARAC[ hpName ]
        # Get type
        hpType = hpCarac[ 0 ]
        
        if ( hpType == "int" ) :
            return IntVar()
        elif ( hpType == "float" ) :
            return DoubleVar()
        elif ( hpType == "boolean" ) :
            return BooleanVar()
        else :
            raise ValueError( "Unknown type", hpType )

        
class CreateOrUpdateConfigWindow ( MyDialog ) :
    
    def __init__( self, boss, callbackFct, **options ) :
        MyDialog.__init__( self, boss, callbackFct, **options )
 
    def run( self, config = None ) :
        "Create or Update view"
    
        frameTop = Frame( self, relief=GROOVE )
        frameTop.pack(side=TOP, padx=30, pady=30)
    
        label = Label( frameTop, text="Configuration form" )
        label.pack()
    
        frameForm = Frame( self, relief=GROOVE )
        frameForm.pack( padx=30, pady=30)
    
        frameButtons = LabelFrame( self, borderwidth=0 )
        frameButtons.pack( side=BOTTOM, padx=30, pady=30, fill='both', expand=True )
    
        # Normalize form
        if ( config == None ) :
            config = { "name": "", "structure": "" }

        # Table labels
        iRow = 1
        
        for ( key, value ) in config.items() :
    
            # Label value
            # Label
            label = Label( frameForm, text=key, borderwidth=1 ).grid( row=iRow, column=1, sticky=W, padx=10 )
    
            # input : type depends on hp name
            inputVar = StringVar()
            
            input = Entry( frameForm, textvariable=inputVar ).grid( row=iRow, column=2, sticky=W, padx=10 )
            self.inputs[ key ] = inputVar

            inputVar.set( value )
            
            iRow += 1
    
        buttonSave   = Button( frameButtons, text="Save"  , command=self.buttonOkClicked    , state=NORMAL )
        buttonCancel = Button( frameButtons, text="Cancel", command=self.buttonCancelClicked, state=NORMAL )
        
        buttonSave.pack( side=LEFT, padx=40 )
        buttonCancel.pack( side=RIGHT, padx=40 )

