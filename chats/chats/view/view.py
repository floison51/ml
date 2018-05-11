'''
Created on 28 avr. 2018
Machine Learning configuration
@author: fran
'''
import tkinter as tk

buttonsToUpdate = []

def buildConfigGrid( frameConfigs, configs, guiDoer ):

#     frameConfigsGlobalTable = tk.LabelFrame( frameConfigs, padx=10, pady=10)
#     frameConfigsGlobalTable.pack( side = "top" )
#
#     scrollbar = tk.Scrollbar( frameConfigsGlobalTable )
#     scrollbar.pack( side = tk.RIGHT, fill = tk.Y )

#     frameConfigsTable = tk.LabelFrame( frameConfigsGlobalTable )

    frameConfigsTable = tk.LabelFrame( frameConfigs, padx=10, pady=10)
    frameConfigsTable.pack( side = "top" )

    # show window
    confSelected = tk.StringVar()

    labels = { 1: "", 2: "Id", 3: "Name", 4: "Structure", 5 : "Best DEV Accuracy", 6: "Hyper Params" }

    for iCol in range( 1, 7 ) :
        label = tk.Label( frameConfigsTable, text=labels[ iCol ], borderwidth=1 ).grid( row=1, column=iCol )

    iRow = 2
    for config in configs :

        idConf = config[ 0 ]

        # Radio button
        radioButton = tk.Radiobutton( frameConfigsTable, variable=confSelected, value=idConf, command=confRbClicked ).grid( row=iRow, column=1 )

        iCol = 2
        for item in config:
            label = tk.Label( frameConfigsTable, text=item, borderwidth=1 ).grid( row=iRow, column=iCol )
            iCol += 1

        # Button to show hyper params
        buttonShowHP = tk.Button( frameConfigsTable, text="Show", command=(lambda s=idConf : guiDoer.showHyperParams( s )) ).grid( row=iRow, column=iCol )

        iRow += 1

    frameConfigsButtons = tk.LabelFrame( frameConfigs, padx=10, pady=10)
    frameConfigsButtons.pack( side = "bottom" )

    buttonNewConfig = tk.Button( frameConfigsButtons, text="New", command=guiDoer.newConfig )
    buttonNewConfig.grid( row=1, column=1 )
    buttonUpdateConfig = tk.Button( frameConfigsButtons, text="Update", command=guiDoer.updateConfig, state=tk.DISABLED )
    buttonUpdateConfig.grid( row=1, column=2 )
    buttonDeleteConfig = tk.Button( frameConfigsButtons, text="Delete", command=guiDoer.deleteConfig, state=tk.DISABLED )
    buttonDeleteConfig.grid( row=1, column=3 )

    buttonsToUpdate.append( buttonUpdateConfig )
    buttonsToUpdate.append( buttonDeleteConfig )
    # Last one is a button
    #button = Button(frameConfigs, text=labels[ len( labels ) ], borderwidth=1).grid( row=1, column=len( labels ) )
    #button.pck()

def enableEntry( entry ):
    entry.configure( state=tk.NORMAL )
    entry.update()

def confRbClicked():
    for button in buttonsToUpdate :
        enableEntry( button )

def showMainWindow( configs, guiDoer ):
    fenetre = tk.Tk()

    frameTop = tk.Frame( fenetre, relief=tk.GROOVE )
    frameTop.pack(side=tk.TOP, padx=30, pady=30)

    label = tk.Label( frameTop, text="Cat's Recognition Machine Learning" )
    label.pack()

    frameConfigs = tk.LabelFrame(fenetre, text="Configurations", padx=20, pady=20)
    buildConfigGrid( frameConfigs, configs, guiDoer )
    frameConfigs.pack(fill="both", expand="yes", padx=10, pady=10)

    frameButtons = tk.Frame(fenetre, borderwidth=2, relief=tk.GROOVE)
    frameButtons.pack( side=tk.BOTTOM, padx=30, pady=30)

    frameConfigs = tk.Frame(fenetre, borderwidth=2, relief=tk.GROOVE)
    frameConfigs.pack(side=tk.LEFT, padx=30, pady=30)

    buttonRun = tk.Button( frameButtons, text="Run", command=fenetre.quit, state=tk.DISABLED )
    buttonRun.pack( side="left" )

    buttonsToUpdate.append( buttonRun )

    buttonCancel=tk.Button( frameButtons, text="Cancel", command=fenetre.quit)
    buttonCancel.pack( side="right")

    fenetre.mainloop()

class ViewOrUpdateHyperParamsUpdateWindow :
    
    def __init__( self ) :
        self.result = None
        self.inputs = {}
        self.fenetre = tk.Tk()

    def run( self, dbHyperParams ) :
        "View or Update hyper parameters"
    
        frameTop = tk.Frame( self.fenetre, relief=tk.GROOVE )
        frameTop.pack(side=tk.TOP, padx=30, pady=30)
    
        label = tk.Label( frameTop, text="Hyper parameter form" )
        label.pack()
    
        frameForm = tk.Frame( self.fenetre, relief=tk.GROOVE )
        frameForm.pack( padx=30, pady=30)
    
        frameButtons = tk.Frame(self.fenetre, borderwidth=2, relief=tk.GROOVE)
        frameButtons.pack( side=tk.BOTTOM, padx=30, pady=30)
    
        # Add dico entries
        iRow = 1
    
        #Get real hyper params
        hyperParams = dbHyperParams[ "hyper_params" ]
        
        for key, value in hyperParams.items() :
    
            # Label
            label = tk.Label( frameForm, text=key, borderwidth=1 ).grid( row=iRow, column=1 )
    
            # input
            inputVar = tk.StringVar()
            input = tk.Entry( frameForm, textvariable=inputVar ).grid( row=iRow, column=2 )
            self.inputs[ key ] = inputVar

            inputVar.set( str( value ) )
    
            iRow += 1
    
        buttonSave   = tk.Button( frameButtons, text="Save"  , command=self.buttonOkClicked    , state=tk.NORMAL )
        buttonCancel = tk.Button( frameButtons, text="Cancel", command=self.buttonCancelClicked, state=tk.NORMAL )
        
        buttonSave.pack()
        buttonCancel.pack()
    
        # Run window
        ## TODO modal
        self.fenetre.mainloop()
        
        return self.result

    def buttonCancelClicked( self ) : 
        # Bye
        result = None
        self.fenetre.destroy()
        
    def buttonOkClicked( self ) : 
        # Bye
        print( self.inputs )
        for key, value in self.inputs.items() :
            print( key, "=", value.get() )
        result = None
        self.fenetre.destroy()
        