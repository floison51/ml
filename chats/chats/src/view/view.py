'''
Created on 28 avr. 2018
Machine Learning configuration
@author: fran
'''
from tkinter import *
from tkinter.ttk import Combobox

import const.constants as const

buttonsToUpdate = []

class MainWindow ( Tk ):

    def __init__( self, configDoer, hpDoer, runsDoer ) :

        Tk.__init__(self)

        self.configDoer = configDoer
        self.hpDoer = hpDoer
        self.runsDoer = runsDoer

        # From conf radio-buttons
        self.confSelected = IntVar()
        self.confSelected.set( -1 )

        #Nb rows added
        self.nbRowsAdded = 0

        self.buttonClicked = None
        self.runParams = None

    def showAndSelectConf( self, configs ):

        frameTop = Frame( self, relief=GROOVE )
        frameTop.pack(side=TOP, padx=30, pady=30)

        label = Label( frameTop, text="Cat's Recognition Machine Learning" )
        label.pack()

        frameConfigs = LabelFrame( self, text="Configurations", padx=20, pady=20)
        self.buildConfigGrid( frameConfigs, configs )
        frameConfigs.pack(fill="both", expand="yes", padx=10, pady=10)

        frameButtons = Frame(self, borderwidth=0 )
        frameButtons.pack( side=BOTTOM, padx=10, pady=10, fill='both', expand=True )

        buttonTrain = Button(
            frameButtons, text="Train",
            command=lambda name="Train" : ( self.clickButton( name ) ),
            state=DISABLED
        )
        buttonTrain.pack( side="left", padx=40 )
        buttonsToUpdate.append( buttonTrain )

        buttonPredict = Button(
            frameButtons, text="Predict",
            command=lambda name="Predict" : ( self.clickButton( name ) ),
            state=DISABLED
        )
        buttonPredict.pack( side="left", padx=40 )
        buttonsToUpdate.append( buttonPredict )

        buttonCancel=Button(
            frameButtons, text="Cancel",
            command=lambda name="Cancel" : ( self.clickButton( name ) ),
        )
        buttonCancel.pack( side="right", padx=40  )

        self.mainloop()

        # return selected idConf
        return ( self.confSelected.get(), self.buttonClicked, self.runParams )


    def buildConfigGrid( self, frameConfigs, configs ):

        #Configuration rows
        self.rows = {}
        self.rowVarLabels = {}

        self.frameConfigsTable = LabelFrame( frameConfigs, padx=10, pady=10)

        # show window
        labels = { 1: "", 2: "Id", 3: "Name", 4: "Machine", 5: "Image Size", 6 : "Structure", 7 : "Best DEV\nAccuracy", 8: "Hyper Params", 9: "Runs" }

        for iCol in range( 1, len( labels ) + 1 ) :
            label = Label( self.frameConfigsTable, text=labels[ iCol ], borderwidth=1 ).grid( row=0, column=iCol, sticky=W, padx=10 )

        idConf = None

        for config in configs :
            self.addRowConfig( config )

        frameConfigsButtons = LabelFrame( frameConfigs, padx=10, pady=10, borderwidth=0 )
        frameConfigsButtons.pack( side = "bottom", fill='both', expand=True )

        buttonNewConfig = Button( \
            frameConfigsButtons, text="New", \
            command=lambda : ( self.configDoer.createConfig( self ) ) \
        )
        buttonNewConfig.grid( row=1, column=1, padx=10 )

        buttonUpdateConfig = Button( \
            frameConfigsButtons, text="Update", \
            command=lambda idConf=idConf : ( self.configDoer.updateConfig( self, self.confSelected.get() ) ), \
            state=DISABLED
        )
        buttonUpdateConfig.grid( row=1, column=2, padx=10 )

        buttonDeleteConfig = Button( \
            frameConfigsButtons, text="Delete",
            command=lambda idConf=idConf : ( self.configDoer.deleteConfig( self, self.confSelected.get() ) ), \
            state=DISABLED \
        )
        buttonDeleteConfig.grid( row=1, column=3, padx=10 )

        buttonsToUpdate.append( buttonUpdateConfig )
        buttonsToUpdate.append( buttonDeleteConfig )

        self.frameConfigsTable.pack( side = "top" )

    def addRowConfig( self, config ):

        colNames = const.ConfigsDico.DISPLAY_FIELDS

        cols = []
        colVarLabels = []

        idConf = config[ "id" ]
        iRow = self.nbRowsAdded + 1 # 1 for header

        # Radio button
        radioButton = Radiobutton( \
            self.frameConfigsTable, variable=self.confSelected, value=idConf, \
            command=self.confRbClicked
        )
        cols.append( radioButton )
        radioButton.grid( row=iRow, column=1 )

        iCol = 2
        for colName in colNames:

            item = config[ colName ]

            # format?
            formatString = const.ConfigsDico.CARAC[ colName ][ 2 ]
            if ( ( item != None ) and ( formatString != None ) ) :
                item = formatString.format( item )

            if ( colName == "structure" ) :
                # special case: show button

                # Button to show structure
                buttonShowStructure = Button( \
                    self.frameConfigsTable, text="Show", \
                    command= lambda idConf=idConf : self.configDoer.showStructure( self, idConf )
                )
                cols.append( buttonShowStructure )
                buttonShowStructure.grid( row=iRow, column=iCol )

            else :
                ## We need a var of label values, to be able to modify labels
                varLabel = getInputVar( const.ConfigsDico.CARAC, colName )

                varLabel.set( item )
                colVarLabels.append( varLabel )

                label = Label( self.frameConfigsTable, borderwidth=1, textvariable=varLabel )

                cols.append( label )
                label.grid( row=iRow, column=iCol, sticky=W, padx=10 )

            iCol += 1

        # Button to show hyper params
        buttonShowHP = Button( \
            self.frameConfigsTable, text="Update", \
            command= lambda idConf=idConf : self.hpDoer.updateHyperParams( self, idConf )
        )
        cols.append( buttonShowHP )
        buttonShowHP.grid( row=iRow, column=iCol )
        iCol += 1

        # Button to show runs
        buttonShowRuns = Button( \
            self.frameConfigsTable, text="Show", \
            command= lambda idConf=idConf : self.runsDoer.showRuns( self, idConf )
        )
        cols.append( buttonShowRuns )
        buttonShowRuns.grid( row=iRow, column=iCol )

        ## Declare row
        self.rows[ idConf ] = cols
        self.rowVarLabels[ idConf ] = colVarLabels;

        ## Count
        self.nbRowsAdded += 1

    def updateRowConfig( self, config ):

        colNames = const.ConfigsDico.DISPLAY_FIELDS

        ## get conf id
        id = config[ "id" ]
        varLabels = self.rowVarLabels[ id ]

        iCol = 0 # 1 = offset radio-button
        for colName in colNames:
            
            if ( colName != "structure" ) :
                # associated var
                var = varLabels[ iCol ]
                # get and set value
                value  = config[ colName ]
                var.set( value )
                
                iCol += 1

    def deleteConfigGrid( self, idConfig ):
        cols = self.rows[ idConfig ]
        for col in cols :
            col.destroy()
        # delete line
        self.rows.pop( idConfig )
        # Deselect main buttons
        for button in buttonsToUpdate :
            self.disableEntry( button )

    def addConfigGrid( self, config ):
        self.addRowConfig( config )

    def updateConfigGrid( self, config ):
        self.updateRowConfig( config )

    def enableEntry( self, entry ):
        entry.configure( state=NORMAL )
        entry.update()

    def disableEntry( self, entry ):
        entry.configure( state=DISABLED )
        entry.update()

    def confRbClicked( self ):
        for button in buttonsToUpdate :
            self.enableEntry( button )

    def clickButton( self, buttonName ):
        self.buttonClicked = buttonName

        # Next dialog
        if ( self.buttonClicked == "Train" ) :
            startTrainingDialog = StartTrainDialog( self, self.doRunTraining )
            startTrainingDialog.run()
        else :
            self.destroy()

    def doRunTraining( self ):
        self.destroy()


class MyDialog( Toplevel ):
    "Modal dialog window"

    def __init__( self, boss, callbackFct, **options ) :
        Toplevel.__init__( self, boss, **options )

        # Modal window
        ## disable parent window
        boss.wm_attributes( "-disabled", True )
        self.transient( boss )

        self.callbackFct = callbackFct
        self.result = None
        self.inputs = {}

    def close( self ):
        "Override close to deleselct modal mode"
        ## enable parent window
        self.master.wm_attributes( "-disabled", False )
        super( MyDialog, self ).close()

    def destroy( self ):
        "Override close to deleselct modal mode"
        ## enable parent window
        self.master.wm_attributes( "-disabled", False )
        super( MyDialog, self ).destroy()

    def do_close( self ) :
        ## enable parent window
        self.master.wm_attributes( "-disabled", False )
        self.destroy()
        # update callback
        self.callbackFct( self, self.result )

    def buttonCancelClicked( self ) :
        self.buttonClicked = "cancel"
        # Bye
        self.result = None
        self.do_close()

    def buttonOkClicked( self ) :

        # Read values from form
        self.result = {}
        for key, value in self.inputs.items() :

            if ( isinstance( value, Text ) ) :
                # special set for text
                result = value.get( "1.0", END )
            else :
                #Normal var
                result = value.get()

            self.result[ key ] = result

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

        # Format accuracy
        strFormat = const.ConfigsDico.CARAC[ "bestAccuracy" ][ 2 ]
        formattedBestDevAccuracy = strFormat.format( bestDevAccuracy )

        labelText1 = "Current\nhyper params"
        labelText2 = "Best\nhyper params\nDEV accuracy=" + formattedBestDevAccuracy
        Label( frameForm, text=labelText1, borderwidth=1 ).grid( row=iRow, column=1, sticky=W, padx=10 )
        Label( frameForm, text=labelText2, borderwidth=1 ).grid( row=iRow, column=3, sticky=W, padx=10 )

        # Table labels
        iRow += 1

        for key in hyperParams.keys() :

            # Label value
            # Label
            label = Label( frameForm, text=key, borderwidth=1 ).grid( row=iRow, column=1, sticky=W, padx=10 )

            # input : type depends on hp name
            inputVar = getInputVar( const.HyperParamsDico.CARAC, key )

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

class CreateOrUpdateConfigWindow ( MyDialog ) :

    def __init__( self, boss, callbackFct, **options ) :
        MyDialog.__init__( self, boss, callbackFct, **options )

    def run( self, machineNames, config = None ) :
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
            config = {
                "name": "", "structure": "",
                "imageSize" : const.ConfigsDico.CARAC[ "imageSize" ][ 1 ],
                "machine"   : const.MachinesDico.CARAC[ "name" ][ 1 ]
            }

        # Table labels
        iRow = 1

        for ( key, value ) in config.items() :

            # IDs are hidden fileds
            isKey = ( key in const.ConfigsDico.KEYS )

            # Label value
            if isKey :
                inputVar = IntVar()
            else :

                # input : type depends on hp name
                inputVar = StringVar()

                label = Label( frameForm, text=key, borderwidth=1 ).grid( row=iRow, column=1, sticky=W, padx=10 )

                # Special case for machines
                if ( key == "structure" ) :
                    ## special frame
                    frameEdit = Frame( frameForm, relief=GROOVE )
                    # text editor
                    text = Text( frameEdit, width=80, height=10 )
                    scroll = Scrollbar( frameEdit, command=text.yview )
                    text.configure( yscrollcommand=scroll.set )

                    text.pack(side=LEFT)
                    scroll.pack(side=RIGHT, fill=Y)

                    frameEdit.grid( row=iRow, column=2, sticky=W, padx=10 )

                    inputVar = text

                # Special case for structure : edit text
                elif ( key == "machine" ) :
                    # Combo box
                    Combobox( frameForm, textvariable = inputVar, \
                        values = machineNames, state = 'readonly').grid( row=iRow, column=2, sticky=W, padx=10 )
                else :
                    # Label
                    input = Entry( frameForm, textvariable=inputVar   ).grid( row=iRow, column=2, sticky=W, padx=10 )

            self.inputs[ key ] = inputVar

            # set value
            if ( isinstance( inputVar, Text ) ) :
                # special set for text
                inputVar.insert( INSERT, value )
            else :
                #Normal var
                inputVar.set( value )

            iRow += 1

        buttonSave   = Button( frameButtons, text="Save"  , command=self.buttonOkClicked    , state=NORMAL )
        buttonCancel = Button( frameButtons, text="Cancel", command=self.buttonCancelClicked, state=NORMAL )

        buttonSave.pack( side=LEFT, padx=40 )
        buttonCancel.pack( side=RIGHT, padx=40 )


def getInputVar( dicoCarac, hpName ) :
    # use dico
    hpCarac = dicoCarac[ hpName ]
    # Get type
    hpType = hpCarac[ 0 ]

    if ( hpType == "int" ) :
        return IntVar()
    elif ( hpType == "string" ) :
        return StringVar()
    elif ( hpType == "float" ) :
        return DoubleVar()
    elif ( hpType == "boolean" ) :
        return BooleanVar()
    else :
        raise ValueError( "Unknown type", hpType )

class TextModalWindow ( MyDialog ) :

    def __init__( self, boss, callbackFct, **options ) :
        MyDialog.__init__( self, boss, callbackFct, **options )

    def run( self, structure, readOnly ) :
        "Modal text window"

        frameEdit = Frame( self, relief=GROOVE )
        frameEdit.pack(side=TOP, padx=30, pady=30)

        # text editor
        text = Text( frameEdit, width=80, height=10 )
        scroll = Scrollbar( frameEdit, command=text.yview )
        text.configure( yscrollcommand=scroll.set )

        text.pack(side=LEFT)
        scroll.pack(side=RIGHT, fill=Y)

        text.insert( "1.0", structure )

        #Read only
        text.config( state=DISABLED )

class StartTrainDialog( MyDialog ):

    def __init__( self, boss, callbackFct, **options ) :
        MyDialog.__init__( self, boss, callbackFct, **options )

    def run( self ) :

        frameTop = Frame( self, relief=GROOVE )
        frameTop.pack(side=TOP, padx=30, pady=30)

        label = Label( frameTop, text="Start training form" )
        label.pack()

        frameForm = Frame( self, relief=GROOVE )
        frameForm.pack( padx=30, pady=30)

        frameButtons = LabelFrame( self, borderwidth=0 )
        frameButtons.pack( side=BOTTOM, padx=30, pady=30, fill='both', expand=True )

        Label( frameForm, text="Run comment", borderwidth=1 ).grid( row=1, column=1, sticky=W, padx=10 )
        self.commentInputVar = getInputVar( const.RunsDico.CARAC, "comment" )
        Entry( frameForm, textvariable=self.commentInputVar, width = 40 ).grid( row=1, column=2, sticky=W, padx=10 )

        Label( frameForm, text="Tune", borderwidth=1 ).grid( row=2, column=1, sticky=W, padx=10 )
        self.tuneInputVar = BooleanVar()
        Checkbutton( frameForm, variable=self.tuneInputVar ).grid( row=2, column=2, sticky=W, padx=10 )

        Label( frameForm, text="Nb tuning cycles", borderwidth=1 ).grid( row=3, column=1, sticky=W, padx=10 )
        self.nbTuningInputVar = IntVar()
        self.nbTuningInputVar.set( 20 )
        Entry( frameForm, textvariable=self.nbTuningInputVar ).grid( row=3, column=2, sticky=W, padx=10 )

        buttonRun    = Button( frameButtons, text="Run"   , command=self.buttonRunClicked   , state=NORMAL )
        buttonCancel = Button( frameButtons, text="Cancel", command=self.buttonCancelClicked, state=NORMAL )

        buttonRun.pack( side=LEFT, padx=40 )
        buttonCancel.pack( side=RIGHT, padx=40 )

    def buttonRunClicked( self ) :
        # Give params to master
        self.master.runParams = { 
            "comment"   : self.commentInputVar.get(), 
            "tune"      : self.tuneInputVar.get(),
            "nbTuning"  : self.nbTuningInputVar.get() 
        }
        self.destroy()
        self.master.destroy()

    def buttonCancelClicked( self ) :
        # cancel
        self.master.clickButton( "Cancel" )
        self.destroy()
        self.master.destroy()
