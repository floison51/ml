'''
Created on 28 avr. 2018
Machine Learning configuration
@author: fran
'''
from tkinter import *

buttonsToUpdate = []

def buildConfigGrid( frameConfigs, configs, guiDoer ):

#     frameConfigsGlobalTable = LabelFrame( frameConfigs, padx=10, pady=10)
#     frameConfigsGlobalTable.pack( side = "top" )
#
#     scrollbar = Scrollbar( frameConfigsGlobalTable )
#     scrollbar.pack( side = RIGHT, fill = Y )

#     frameConfigsTable = LabelFrame( frameConfigsGlobalTable )

    frameConfigsTable = LabelFrame( frameConfigs, padx=10, pady=10)
    frameConfigsTable.pack( side = "top" )

    # show window
    confSelected = StringVar()

    labels = { 1: "", 2: "Id", 3: "Name", 4: "Structure", 5 : "Best DEV Accuracy", 6: "Hyper Params" }

    for iCol in range( 1, 7 ) :
        label = Label( frameConfigsTable, text=labels[ iCol ], borderwidth=1 ).grid( row=1, column=iCol )

    iRow = 2
    for config in configs :

        idConf = config[ 0 ]

        # Radio button
        radioButton = Radiobutton( frameConfigsTable, variable=confSelected, value=idConf, command=confRbClicked ).grid( row=iRow, column=1 )

        iCol = 2
        for item in config:
            label = Label( frameConfigsTable, text=item, borderwidth=1 ).grid( row=iRow, column=iCol )
            iCol += 1

        # Button to show hyper params
        buttonShowHP = Button( frameConfigsTable, text="Show", command=(lambda s=idConf : guiDoer.showHyperParams( s )) ).grid( row=iRow, column=iCol )

        iRow += 1

    frameConfigsButtons = LabelFrame( frameConfigs, padx=10, pady=10)
    frameConfigsButtons.pack( side = "bottom" )

    buttonNewConfig = Button( frameConfigsButtons, text="New", command=guiDoer.newConfig )
    buttonNewConfig.grid( row=1, column=1 )
    buttonUpdateConfig = Button( frameConfigsButtons, text="Update", command=guiDoer.updateConfig, state=DISABLED )
    buttonUpdateConfig.grid( row=1, column=2 )
    buttonDeleteConfig = Button( frameConfigsButtons, text="Delete", command=guiDoer.deleteConfig, state=DISABLED )
    buttonDeleteConfig.grid( row=1, column=3 )

    buttonsToUpdate.append( buttonUpdateConfig )
    buttonsToUpdate.append( buttonDeleteConfig )
    # Last one is a button
    #button = Button(frameConfigs, text=labels[ len( labels ) ], borderwidth=1).grid( row=1, column=len( labels ) )
    #button.pck()

def enableEntry( entry ):
    entry.configure( state=NORMAL )
    entry.update()

def confRbClicked():
    for button in buttonsToUpdate :
        enableEntry( button )

def showMainWindow( configs, guiDoer ):
    fenetre = Tk()

    frameTop = Frame( fenetre, relief=GROOVE )
    frameTop.pack(side=TOP, padx=30, pady=30)

    label = Label( frameTop, text="Cat's Recognition Machine Learning" )
    label.pack()

    frameConfigs = LabelFrame(fenetre, text="Configurations", padx=20, pady=20)
    buildConfigGrid( frameConfigs, configs, guiDoer )
    frameConfigs.pack(fill="both", expand="yes", padx=10, pady=10)

    frameButtons = Frame(fenetre, borderwidth=2, relief=GROOVE)
    frameButtons.pack( side=BOTTOM, padx=30, pady=30)

    frameConfigs = Frame(fenetre, borderwidth=2, relief=GROOVE)
    frameConfigs.pack(side=LEFT, padx=30, pady=30)

    buttonRun = Button( frameButtons, text="Run", command=fenetre.quit, state=DISABLED )
    buttonRun.pack( side="left" )

    buttonsToUpdate.append( buttonRun )

    buttonCancel=Button( frameButtons, text="Cancel", command=fenetre.quit)
    buttonCancel.pack( side="right")

    fenetre.mainloop()

class ViewOrUpdateHyperParamsUpdateWindow ( Toplevel ) :
    
    def __init__( self, **options ) :
        Toplevel.__init__( self, **options )
        # Fenetre modale
        ## TODO : disable parent window
        
        self.transient( self.master )
        
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
    
        frameButtons = Frame( self, borderwidth=2, relief=GROOVE)
        frameButtons.pack( side=BOTTOM, padx=30, pady=30)
    
        # Add dico entries
        iRow = 1
    
        #Get real hyper params
        hyperParams = dbHyperParams[ "hyper_params" ]
        
        for key, value in hyperParams.items() :
    
            # Label
            label = Label( frameForm, text=key, borderwidth=1 ).grid( row=iRow, column=1 )
    
            # input
            inputVar = StringVar()
            input = Entry( frameForm, textvariable=inputVar ).grid( row=iRow, column=2 )
            self.inputs[ key ] = inputVar

            inputVar.set( str( value ) )
    
            iRow += 1
    
        buttonSave   = Button( frameButtons, text="Save"  , command=self.buttonOkClicked    , state=NORMAL )
        buttonCancel = Button( frameButtons, text="Cancel", command=self.buttonCancelClicked, state=NORMAL )
        
        buttonSave.pack()
        buttonCancel.pack()
    
        # Run window
        ## TODO modal
        self.mainloop()
        
        return self.result

    def finish( self ) :
        ## TODO : enable parent window
        self.destroy()
        
    def buttonCancelClicked( self ) : 
        # Bye
        result = None
        self.finish()
        
    def buttonOkClicked( self ) : 
        # Bye
        print( self.inputs )
        for key, value in self.inputs.items() :
            print( key, "=", value.get() )
        result = None
        self.finish()
        