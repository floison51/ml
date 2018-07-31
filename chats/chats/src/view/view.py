'''
Created on 28 avr. 2018
Machine Learning configuration
@author: fran
'''
import os

from tkinter import *
from tkinter.ttk import Combobox
from tkinter.filedialog import askopenfilename

import const.constants as const

buttonsToUpdate = []

class MainWindow ( Tk ):

    def __init__( self, configDoer, hpDoer, runsDoer, startRunDoer, analyzeDoer, predictRunDoer ) :

        Tk.__init__(self)

        self.configDoer = configDoer
        self.hpDoer = hpDoer
        self.runsDoer = runsDoer
        self.startRunDoer   = startRunDoer
        self.analyzeDoer    = analyzeDoer
        self.predictRunDoer = predictRunDoer

        # From conf radio-buttons
        self.confSelected = IntVar()
        self.confSelected.set( -1 )

        #Nb rows added
        self.nbRowsAdded = 0

        self.buttonClicked = None
        self.runParams = None
        self.predictParams = None

        self.varDataset = StringVar()

    def showAndSelectConf( self, conn, datasets, selection ):

        frameTop = Frame( self, relief=GROOVE )
        frameTop.pack(side=TOP, padx=30, pady=30)

        label = Label( frameTop, text="Cat's Recognition Machine Learning" )
        label.pack()

        frameDatasets = LabelFrame( self, text="Data sets", padx=20, pady=20)
        self.buildDatasetFrame( frameDatasets, datasets, selection )
        frameDatasets.pack(fill="both", expand="yes", padx=10, pady=10)

        frameConfigs = LabelFrame( self, text="Configurations", padx=20, pady=20)
        self.buildConfigGrid( conn, frameConfigs, selection )
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

        buttonAnalyse = Button(
            frameButtons, text="Analyze",
            command=lambda name="Analyze" : ( self.clickButton( name ) ),
            state=DISABLED
        )
        buttonAnalyse.pack( side="left", padx=40 )
        buttonsToUpdate.append( buttonAnalyse )

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

        # If a config is selected, eanle buttons
        if ( self.confSelected.get() != -1 ) :
            # Update train, etc... buttons
            self.confRbClicked()


        self.mainloop()

        # save selected things
        selection = {
            "selectedDatasetName" : self.varDataset.get(),
            "selectedIdConfig"    : self.confSelected.get()
        }
        
        import db.db as db
        db.setSelection( conn, const.Selections.KEY_MAIN, selection )
        # commit
        conn.commit()

        # return selected things
        return ( self.varDataset.get(), self.confSelected.get(), self.buttonClicked, self.runParams, self.predictParams )

    def buildDatasetFrame( self, frameDatasets, datasets, selection ):

        datasetNames = []
        for dataset in datasets :
            datasetNames.append( dataset[ "name" ] )

        comboDatasets = Combobox( frameDatasets, textvariable=self.varDataset, values=datasetNames, state="readonly", height=4 )

        if ( len( datasetNames ) > 0 ) :
            # get selected dataset
            if not "selectedDatasetName" in selection :
                # Default selection : first
                selection[ "selectedDatasetName" ] = datasetNames[ 0 ]

            curDatasetName = selection[ "selectedDatasetName" ]
            self.varDataset.set( curDatasetName )

        # Set change listener
        comboDatasets.bind( "<<ComboboxSelected>>", self.updateDataset )
        comboDatasets.pack( side="left", padx=40  )

    def updateDataset( self, event ) :

        ## data set has changed, update config grid
        colVarLabelsByConfigId = self.rowVarLabels;

        # Get configs with best DEV accuracy
        datasetName = self.varDataset.get()
        configs = self.configDoer.getConfigsWithMaxDevAccuracy( datasetName )

        # browse configs
        for config in configs :

            idConfig = config[ "id" ]

            # Get displayed row
            colVarLabels = colVarLabelsByConfigId[ idConfig ]

            # Best DEV accuracy
            item = config[ "bestDevAccuracy" ]
            # format?
            formatString = const.ConfigsDico.CARAC[ "bestDevAccuracy" ][ 2 ]
            if ( ( item != None ) and ( formatString != None ) ) :
                item = formatString.format( item )

            # Update best DEV accuracy
            label = colVarLabels[ 4 ]
            label.set( item )

            # Associated TRN accuracy
            item = config[ "assoTrnAccuracy" ]
            # format?
            formatString = const.ConfigsDico.CARAC[ "assoTrnAccuracy" ][ 2 ]
            if ( ( item != None ) and ( formatString != None ) ) :
                item = formatString.format( item )

            # Update best DEV accuracy
            label = colVarLabels[ 5 ]
            label.set( item )

    def buildConfigGrid( self, conn, frameConfigs, selection ):

        # get dataset
        datasetName = selection[ "selectedDatasetName" ]

        # Get configs
        configs = self.configDoer.getConfigsWithMaxDevAccuracy( datasetName )

        #Configuration rows
        self.rows = {}
        self.rowVarLabels = {}

        self.frameConfigsTable = LabelFrame( frameConfigs, padx=10, pady=10)

        # show window
        labels = {
            1: "", 2: "Id", 3: "Name", 4: "Machine", 5: "Image Size", 6 : "Structure",
            7 : "id Hp", 8: "Hyper Params",
            9 : "Best DEV\nAccuracy", 10: "TRN\nAccuracy",
            11: "last Id Run", 12: "last Run DEV\nAccuracy", 13: "last Run TRN\nAccuracy", 14: "Runs"
        }

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

        # selected config
        if ( "selectedIdConfig" in selection ) :
            selectedIdConfig = selection[ "selectedIdConfig" ]

            if ( selectedIdConfig != None ) :
                self.confSelected.set( selectedIdConfig )

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

            elif ( colName == "json_hperParams" ) :
                # Button to show hyper params
                buttonShowHP = Button( \
                    self.frameConfigsTable, text="Update", \
                    command= lambda idConf=idConf : self.hpDoer.updateHyperParams( self, idConf )
                )
                cols.append( buttonShowHP )
                buttonShowHP.grid( row=iRow, column=iCol )

            else :
                ## We need a var of label values, to be able to modify labels
                varLabel = getInputVar( const.ConfigsDico.CARAC, colName )

                varLabel.set( item )
                colVarLabels.append( varLabel )

                label = Label( self.frameConfigsTable, borderwidth=1, textvariable=varLabel )

                cols.append( label )
                label.grid( row=iRow, column=iCol, sticky=W, padx=10 )

            iCol += 1

        # Button to show runs
        buttonShowRuns = Button( \
            self.frameConfigsTable, text="Show", \
            command= lambda idConf=idConf : self.runsDoer.showRuns( self, self.varDataset.get(), idConf )
        )
        cols.append( buttonShowRuns )
        buttonShowRuns.grid( row=iRow, column=14 )

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

            if ( ( colName != "structure" ) and ( colName != "json_HyperParams" ) ):
                # associated var
                var = varLabels[ iCol ]
                # get and set value
                value  = config[ colName ]
                var.set( value )

                iCol += 1

    def updateRowConfigIdHp( self, idHp ) :

        colNames = const.ConfigsDico.DISPLAY_FIELDS
        idConf = self.confSelected.get()
        varLabels = self.rowVarLabels[ idConf ]

        # get index of idHp
        colIdHp = colNames.index( "idHp" ) - 1 # -1 : radio button
        # associated var
        var = varLabels[ colIdHp ]
        # get and set value
        var.set( idHp )

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
            self.startRunDoer.start( self, self.varDataset.get(), self.confSelected.get() )
        elif ( self.buttonClicked == "Predict" ) :
            self.predictRunDoer.start( self, self.confSelected.get() )
        elif ( self.buttonClicked == "Analyze" ) :
            self.analyzeDoer.analyze( self, self.confSelected.get() )
        else :
            self.destroy()

    def doRunTraining( self ):
        self.destroy()

    def doRunPredict( self ):
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

        #Get real hyper params
        hyperParams     = dbHyperParams[ const.KEY_DICO_HYPER_PARAMS ]
        bestHyperParams = dbBestHyperParams.get( const.KEY_DICO_HYPER_PARAMS, {} )

        # Get data set
        nameDataset = dbHyperParams[ const.KEY_DICO_DATASET_NAME ]

        frameTop = Frame( self, relief=GROOVE )
        frameTop.pack(side=TOP, padx=30, pady=30)

        label = Label( frameTop, text="Hyper parameters" )
        label.pack()
        label = Label( frameTop, text="Data set: " + nameDataset )
        label.pack()

        frameForm = Frame( self, relief=GROOVE )
        frameForm.pack( padx=30, pady=30)

        frameButtons = LabelFrame( self, borderwidth=0 )
        frameButtons.pack( side=BOTTOM, padx=30, pady=30, fill='both', expand=True )

        # Add default values
        for ( key, hpCarac ) in const.HyperParamsDico.CARAC.items() :
            if not key in hyperParams :
                # add default value
                hyperParams[ key ] = hpCarac[ 1 ]

        # Add dico entries
        iRow = 1

        # Format accuracy
        strFormat = const.ConfigsDico.CARAC[ "bestDevAccuracy" ][ 2 ]
        formattedBestDevAccuracy = None
        if ( bestDevAccuracy != None ) :
            formattedBestDevAccuracy = strFormat.format( bestDevAccuracy )

        labelText1 = "Current\nhyper params"
        labelText2 = "Best\nhyper params\nDEV accuracy=" + str( formattedBestDevAccuracy )
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

    return getInputVarForType( hpType )

def getInputVarForType( type ):
    if ( type == "int" ) :
        return IntVar()
    elif ( type == "string" ) :
        return StringVar()
    elif ( type == "float" ) :
        return DoubleVar()
    elif ( type == "boolean" ) :
        return BooleanVar()
    else :
        raise ValueError( "Unknown type", type )

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

    def __init__( self, boss, conn, keySelection, callbackFct, machineFields, machineFieldValues, **options ) :
        MyDialog.__init__( self, boss, callbackFct, **options )
        self.conn = conn
        self.keySelection = keySelection
        self.machineFields = machineFields
        self.machineFieldValues = machineFieldValues

    def run( self ) :

        frameTop = Frame( self, relief=GROOVE )
        frameTop.pack(side=TOP, padx=30, pady=30)

        label = Label( frameTop, text="Start run" )
        label.pack()

        frameForm = Frame( self, relief=GROOVE )
        frameForm.pack( padx=30, pady=30)

        frameButtons = LabelFrame( self, borderwidth=0 )
        frameButtons.pack( side=BOTTOM, padx=30, pady=30, fill='both', expand=True )

        iRow = 1

        Label( frameForm, text="Run comment", borderwidth=1 ).grid( row=iRow, column=1, sticky=W, padx=10 )
        self.commentInputVar = getInputVar( const.RunsDico.CARAC, "comment" )
        self.commentInputVar.set( self.machineFieldValues.get( "comment", "" ) )
        Entry( frameForm, textvariable=self.commentInputVar, width = 40 ).grid( row=iRow, column=2, sticky=W, padx=10 )
        iRow += 1

        Label( frameForm, text="Show plots", borderwidth=1 ).grid( row=iRow, column=1, sticky=W, padx=10 )
        self.showPlotsInputVar = BooleanVar()
        self.showPlotsInputVar.set( self.machineFieldValues.get( "showPlot", True ) )
        Checkbutton( frameForm, variable=self.showPlotsInputVar ).grid( row=iRow, column=2, sticky=W, padx=10 )
        iRow += 1

        Label( frameForm, text="Tune", borderwidth=1 ).grid( row=iRow, column=1, sticky=W, padx=10 )
        self.tuneInputVar = BooleanVar()
        self.tuneInputVar.set( self.machineFieldValues.get( "tune", False ) )
        Checkbutton( frameForm, variable=self.tuneInputVar ).grid( row=iRow, column=2, sticky=W, padx=10 )
        iRow += 1

        Label( frameForm, text="Nb tuning cycles", borderwidth=1 ).grid( row=iRow, column=1, sticky=W, padx=10 )
        self.nbTuningInputVar = IntVar()
        self.nbTuningInputVar.set( self.machineFieldValues.get( "nbTuning", 20 ) )
        Entry( frameForm, textvariable=self.nbTuningInputVar ).grid( row=iRow, column=2, sticky=W, padx=10 )
        iRow += 1

        # Machine fields
        self.inputMachineFields = {}

        for ( key, machineField ) in self.machineFields.items() :
            #Label
            Label( frameForm, text=machineField[ 0 ], borderwidth=1 ).grid( row=iRow, column=1, sticky=W, padx=10 )
            #Input var
            inputVar = getInputVarForType( machineField[ 1 ] )
            #save it
            self.inputMachineFields[ key ] = inputVar

            # Default value
            inputVar.set( self.machineFieldValues.get( key, machineField[ 2 ] ) )
            # field
            if ( machineField[ 1 ] == "boolean" ) :
                Checkbutton( frameForm, variable=inputVar ).grid( row=iRow, column=2, sticky=W, padx=10 )
            else :
                Entry( frameForm, textvariable=inputVar ).grid( row=iRow, column=2, sticky=W, padx=10 )
            iRow += 1

        buttonRun    = Button( frameButtons, text="Run"   , command=self.buttonRunClicked   , state=NORMAL )
        buttonCancel = Button( frameButtons, text="Cancel", command=self.buttonCancelClicked, state=NORMAL )

        buttonRun.pack( side=LEFT, padx=40 )
        buttonCancel.pack( side=RIGHT, padx=40 )
        
        self.mainloop()

    def buttonRunClicked( self ) :
        # Give params to master
        self.master.runParams = {
            "comment"   : self.commentInputVar.get(),
            "showPlots" : self.showPlotsInputVar.get(),
            "tune"      : self.tuneInputVar.get(),
            "nbTuning"  : self.nbTuningInputVar.get(),
        }

        #add machine specific fields
        for ( key, inputVar ) in self.inputMachineFields.items() :
            self.master.runParams[ key ] = inputVar.get()

        # Save selection for next usage
        # save selection DB
        import db.db as db
        db.setSelection( self.conn, self.keySelection, self.master.runParams )
        # commit
        self.conn.commit()
        
        self.destroy()
        self.master.destroy()

    def buttonCancelClicked( self ) :
        # cancel
        self.master.clickButton( "Cancel" )
        self.destroy()

class StartPredictDialog( MyDialog ):

    def __init__( self, boss, callbackFct, **options ) :
        MyDialog.__init__( self, boss, callbackFct, **options )

        self.imagePath = None

    def run( self ) :

        frameTop = Frame( self, relief=GROOVE )
        frameTop.pack(side=TOP, padx=30, pady=30)

        label = Label( frameTop, text="Predict" )
        label.pack()

        frameForm = Frame( self, relief=GROOVE )
        frameForm.pack( padx=30, pady=30)

        frameButtons = LabelFrame( self, borderwidth=0 )
        frameButtons.pack( side=BOTTOM, padx=30, pady=30, fill='both', expand=True )

        iRow = 1

        self.choiceHyperParametersInputVar = getInputVarForType( "int" )
        self.choiceDataInputVar = getInputVarForType( "int" )
        self.photoInputVar = getInputVarForType( "string" )

        Label( frameForm, text="Hyper parameters:", borderwidth=1 ).grid( row=iRow, column=1, sticky=W, padx=10 )
        iRow += 1

        values = [ 1, 2 ]
        labels = [ "Current", "Best" ]

        for i in range( 2 ) :
            Radiobutton(
                frameForm, variable=self.choiceHyperParametersInputVar,
                text = labels[ i ], value=values[ i ],
                command=self.computeEnableRun
            ).grid( row=iRow, column=1, sticky=W, padx=10 )
            iRow += 1

        Label( frameForm, text="Data set:", borderwidth=1 ).grid( row=iRow, column=1, sticky=W, padx=10 )
        iRow += 1

        values = [ 1, 2 ]
        labels = [ "DEV Test set", "Photo" ]

        for i in range( 2 ) :
            Radiobutton(
                frameForm, variable=self.choiceDataInputVar,
                text = labels[ i ], value=values[ i ],
                command=lambda value=values[ i ] : ( self.choiceDataClicked( value ) ),
            ).grid( row=iRow, column=1, sticky=W, padx=10 )
            iRow += 1

        # Photo selection
        self.photo = Entry( frameForm, textvariable=self.photoInputVar, state = DISABLED )
        self.photo.grid( row=iRow - 1, column=3, sticky=W, padx=10 )
        iRow += 1

        #Button to choose image
        self.chooseButton = Button( frameForm, text="Choose", command=self.chooseFile, state=DISABLED )
        self.chooseButton.grid( row=iRow - 1, column=3, sticky=W, padx=10 )

        self.buttonRun    = Button( frameButtons, text="Predict" , command=self.buttonRunClicked   , state=DISABLED )
        buttonCancel = Button( frameButtons, text="Cancel"  , command=self.buttonCancelClicked, state=NORMAL )

        self.buttonRun.pack( side=LEFT, padx=40 )
        buttonCancel.pack( side=RIGHT, padx=40 )

    def chooseFile( self ):

        curDir = os.getcwd().replace( "\\", "/" )
        self.imagePath = askopenfilename(
            initialdir=curDir,
            filetypes = ( ( "Image File", "*.jpg"), ( "Image File", "*.jpeg"), ( "Image File", "*.png"), ("All Files","*.*") ),
            title = "Choose an image."
        )

        basename = os.path.basename( self.imagePath )
        self.photoInputVar.set( basename )

    def choiceDataClicked( self, value ) :
        "Activate photo if needed"
        if ( value == 2 ) :
            # Activate photo input
            self.chooseButton.configure( state = NORMAL )
        else :
            # Disable photo input
            self.chooseButton.configure( state = DISABLED )

        #Check if run button must be enabled
        self.computeEnableRun()

    def computeEnableRun( self ) :
        # Enable predict button?
        if ( ( self.choiceDataInputVar.get() != 0 ) and ( self.choiceHyperParametersInputVar.get() != 0 ) ) :
            self.buttonRun.configure( state = NORMAL )

    def buttonRunClicked( self ) :

        # Give params to master
        self.master.predictParams = {
            "choiceHyperParams" : self.choiceHyperParametersInputVar.get(),
            "choiceData" : self.choiceDataInputVar.get(),
            "imagePath"  : self.imagePath
        }

        self.destroy()
        self.master.destroy()

    def buttonCancelClicked( self ) :
        # cancel
        self.master.clickButton( "Cancel" )
        self.destroy()
