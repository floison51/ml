'''
Created on 28 avr. 2018
Machine Learning configuration
@author: fran
'''
from tkinter import *
import const.constants as const

buttonsToUpdate = []

class MainWindow :
    
    def __init__( self ) :
    
        self.fenetre = Tk()
        # From conf radio-buttons
        self.confSelected = StringVar()
        self.confSelected.set( "None" )
    
    def show( self, configs, guiDoer ):
    
        frameTop = Frame( self.fenetre, relief=GROOVE )
        frameTop.pack(side=TOP, padx=30, pady=30)
    
        label = Label( frameTop, text="Cat's Recognition Machine Learning" )
        label.pack()
    
        frameConfigs = LabelFrame(self.fenetre, text="Configurations", padx=20, pady=20)
        self.buildConfigGrid( frameConfigs, configs, guiDoer )
        frameConfigs.pack(fill="both", expand="yes", padx=10, pady=10)
    
        frameButtons = Frame(self.fenetre, borderwidth=0 )
        frameButtons.pack( side=BOTTOM, padx=10, pady=10, fill='both', expand=True )
    
        buttonRun = Button( frameButtons, text="Run", command=self.fenetre.quit, state=DISABLED )
        buttonRun.pack( side="left", padx=40 )
    
        buttonsToUpdate.append( buttonRun )
    
        buttonCancel=Button( frameButtons, text="Cancel", command=self.fenetre.quit)
        buttonCancel.pack( side="right", padx=40  )
    
        self.fenetre.mainloop()

    def buildConfigGrid( self, frameConfigs, configs, guiDoer ):
    
    #     frameConfigsGlobalTable = LabelFrame( frameConfigs, padx=10, pady=10)
    #     frameConfigsGlobalTable.pack( side = "top" )
    #
    #     scrollbar = Scrollbar( frameConfigsGlobalTable )
    #     scrollbar.pack( side = RIGHT, fill = Y )
    
    #     frameConfigsTable = LabelFrame( frameConfigsGlobalTable )
    
        frameConfigsTable = LabelFrame( frameConfigs, padx=10, pady=10)
        frameConfigsTable.pack( side = "top" )
    
        # show window
        labels = { 1: "", 2: "Id", 3: "Name", 4: "Structure", 5 : "Best DEV\nAccuracy", 6: "Hyper Params" }
    
        for iCol in range( 1, 7 ) :
            label = Label( frameConfigsTable, text=labels[ iCol ], borderwidth=1 ).grid( row=1, column=iCol, sticky=W, padx=10 )
    
        iRow = 2
        for config in configs :
    
            idConf = config[ 0 ]
    
            # Radio button
            radioButton = Radiobutton( frameConfigsTable, variable=self.confSelected, value=idConf, command=self.confRbClicked ).grid( row=iRow, column=1 )
    
            iCol = 2
            for item in config:
                label = Label( frameConfigsTable, text=item, borderwidth=1 ).grid( row=iRow, column=iCol, sticky=W, padx=10 )
                iCol += 1
    
            # Button to show hyper params
            buttonShowHP = Button( frameConfigsTable, text="Show", command=(lambda idConf=idConf : guiDoer.showHyperParams( self.fenetre, idConf )) ).grid( row=iRow, column=iCol )
    
            iRow += 1
    
        frameConfigsButtons = LabelFrame( frameConfigs, padx=10, pady=10, borderwidth=0 )
        frameConfigsButtons.pack( side = "bottom", fill='both', expand=True )
    
        buttonNewConfig = Button( frameConfigsButtons, text="New", command=guiDoer.newConfig )
        buttonNewConfig.grid( row=1, column=1, padx=10 )
        buttonUpdateConfig = Button( frameConfigsButtons, text="Update", command=guiDoer.updateConfig, state=DISABLED )
        buttonUpdateConfig.grid( row=1, column=2, padx=10 )
        buttonDeleteConfig = Button( frameConfigsButtons, text="Delete", command=guiDoer.deleteConfig, state=DISABLED )
        buttonDeleteConfig.grid( row=1, column=3, padx=10 )
    
        buttonsToUpdate.append( buttonUpdateConfig )
        buttonsToUpdate.append( buttonDeleteConfig )
        # Last one is a button
        #button = Button(frameConfigs, text=labels[ len( labels ) ], borderwidth=1).grid( row=1, column=len( labels ) )
        #button.pck()
    
    def enableEntry( self, entry ):
        entry.configure( state=NORMAL )
        entry.update()
    
    def confRbClicked( self ):
        for button in buttonsToUpdate :
            self.enableEntry( button )
    
class ViewOrUpdateHyperParamsUpdateWindow ( Toplevel ) :
    
    def __init__( self, boss, callbackFct, **options ) :
        Toplevel.__init__( self, boss, **options )
        
        # Modal window
        ## disable parent window
        boss.wm_attributes( "-disabled", True )
        self.transient( boss )
        
        self.callbackFct = callbackFct
        self.result = None
        self.inputs = {}
 
    def run( self, dbHyperParams ) :
        "View or Update hyper parameters"
    
        frameTop = Frame( self, relief=GROOVE )
        frameTop.pack(side=TOP, padx=30, pady=30)
    
        label = Label( frameTop, text="Hyper parameter form" )
        label.pack()
    
        frameForm = Frame( self, relief=GROOVE )
        frameForm.pack( padx=30, pady=30)
    
        frameButtons = LabelFrame( self, borderwidth=0 )
        frameButtons.pack( side=BOTTOM, padx=30, pady=30, fill='both', expand=True )
    
        # Add dico entries
        iRow = 1
    
        #Get real hyper params
        hyperParams = dbHyperParams[ "hyper_params" ]
        
        for key, value in hyperParams.items() :
    
            # Label
            label = Label( frameForm, text=key, borderwidth=1 ).grid( row=iRow, column=1, sticky=W, padx=10 )
    
            # input : type depends on hp name
            inputVar = self.getInputVar( key )
            
            input = Entry( frameForm, textvariable=inputVar ).grid( row=iRow, column=2, sticky=W, padx=10 )
            self.inputs[ key ] = inputVar

            inputVar.set( str( value ) )
    
            iRow += 1
    
        buttonSave   = Button( frameButtons, text="Save"  , command=self.buttonOkClicked    , state=NORMAL )
        buttonCancel = Button( frameButtons, text="Cancel", command=self.buttonCancelClicked, state=NORMAL )
        
        buttonSave.pack( side=LEFT, padx=40 )
        buttonCancel.pack( side=RIGHT, padx=40 )
        
    

    def getInputVar( self, hpName ) :
        # use dico
        hpCarac = const.HyperParamsDico.carac[ hpName ]
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

        
    def do_close( self ) :
        ## enable parent window
        self.master.wm_attributes( "-disabled", False )
        self.destroy()
        # update callback
        self.callbackFct( self.result )
        
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
        