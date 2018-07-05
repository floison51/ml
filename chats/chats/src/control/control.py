import view.view as view
from view.viewruns import ViewRunsWindow
import db.db as db
import const.constants as const

class Doer:
    def __init__( self, conn ) :
        self.conn = conn


class HyperParamsDoer( Doer ):

    def __init__( self, conn ) :
        super().__init__( conn )

    def updateHyperParams( self, fenetre, idConf ):

        # get idDataset
        nameDataset = fenetre.varDataset.get()
        idDataset = db.getDatasetIdByName( self.conn, nameDataset )

        # Get config
        self.config = db.getConfig( self.conn, idConf )

        # get hyper parameters
        hyperParams = db.getHyperParams( self.conn, idDataset, idConf )

        # get best hyper params
        ( bestHyperParams, bestDevAccuracy, _ ) = db.getBestHyperParams( self.conn, idDataset, idConf )

        # Launch window, it may update hps
        viewHp = view.ViewOrUpdateHyperParamsWindow( fenetre, self.doCreateOrUpdateHyperParams )

        # launch view with callback
        viewHp.run( hyperParams, bestHyperParams, bestDevAccuracy )

    def doCreateOrUpdateHyperParams( self, fenetre, newHyperParams ):

        print( "Result:", newHyperParams )

        if ( newHyperParams == None ) :
            ## Nothing to do
            return

        # get idDataset
        nameDataset = fenetre.master.varDataset.get()
        idDataset = db.getDatasetIdByName( self.conn, nameDataset )

        # Get or create new hyper parameters
        db.getOrCreateHyperParams( self.conn, idDataset, self.config[ "id" ], newHyperParams )

        #commit
        self.conn.commit()

class ConfigDoer( Doer ):

    def __init__( self, conn ) :
        super().__init__( conn )

    def getConfigsWithMaxDevAccuracy( self, datasetName ):

        # Get idDataset from its name
        idDataset = db.getDatasetIdByName( self.conn, datasetName )

        # get configs
        configs = db.getConfigsWithMaxDevAccuracy( self.conn, idDataset )

        return configs

    def createConfig( self, fenetre ):
        "Create new config"
        viewConfig = view.CreateOrUpdateConfigWindow( fenetre, self.doCreateConfig )

        #Get machine names
        machineNames = db.getMachineNames( self.conn )

        # launch view with callback
        viewConfig.run( machineNames )

    def updateConfig( self, fenetre, idConf ):
        "Update new config"

        # Get config
        config = db.getConfig( self.conn, idConf )

        viewConfig = view.CreateOrUpdateConfigWindow( fenetre, self.doUpdateConfig )

        #Get machine names
        machineNames = db.getMachineNames( self.conn )

        # Add machine name
        idMachine = config[ "idMachine" ]
        machineName = db.getMachineNameById( self.conn, idMachine )
        config[ "machine" ] = machineName

        # launch view with callback
        viewConfig.run( machineNames, config )

    def deleteConfig( self, fenetre, idConf ):
        print( "deleteConfig", idConf )
        db.deleteConfig( self.conn, idConf )

        # Update window
        fenetre.deleteConfigGrid( idConf )

        # commit
        self.conn.commit()

    def doCreateConfig( self, fenetre, newConfig ):

        if ( newConfig == None ) :
            ## Nothing to do
            return

        # Default hyper params
        hyperParams = {}
        for ( key, hpCarac ) in const.HyperParamsDico.CARAC.items() :
            hyperParams[ key ] = hpCarac[ 1 ]

        # normalize structure
        structure = newConfig[ "structure" ].strip()

        # get idDataset
        nameDataset = fenetre.master.varDataset.get()
        idDataset = db.getDatasetIdByName( self.conn, nameDataset )

        # Get or create new config
        idNewConfig = db.createConfig( self.conn, \
            idDataset,
            newConfig[ "name" ], structure, \
            newConfig[ "imageSize" ], \
            newConfig[ "machine" ], hyperParams \
        )

        # Update window
        config = db.getConfigsWithMaxDevAccuracy( self.conn, idDataset, idNewConfig )[ 0 ]
        fenetre.master.addConfigGrid( config )

        # commit
        self.conn.commit()

    def doUpdateConfig( self, fenetre, newConfig ):

        if ( newConfig == None ) :
            ## Nothing to do
            return

        # Convert machine name to machine id
        idMachine = db.getIdMachineByName( self.conn, newConfig[ "machine" ] )
        newConfig[ "idMachine" ] = idMachine

        # normalize structure
        structure = newConfig[ "structure" ].strip()
        newConfig[ "structure" ] = structure

        #Update config
        db.updateConfig( self.conn, newConfig )

        # Update window
        config = db.getConfigsWithMaxDevAccuracy( self.conn, newConfig[ "id" ] )[ 0 ]
        fenetre.master.updateConfigGrid( config )

        # commit
        self.conn.commit()

    def showStructure( self, fenetre, idConf ):

        # Get config
        config = db.getConfig( self.conn, idConf )

        dialogShowStructure = view.TextModalWindow( fenetre, None )
        dialogShowStructure.run( config[ "structure" ], True )

class RunsDoer( Doer ):

    def __init__( self, conn ) :
        super().__init__( conn )

    def showRuns( self, fenetre, datasetName, idConf ):

        if ( idConf == None ) :
            ## Nothing to do
            return

        # Launch window, without callback
        viewRuns = ViewRunsWindow( fenetre, None )

        # get runs
        idDataset = db.getDatasetIdByName( self.conn, datasetName )
        runs = db.getRuns( self.conn, idDataset, idConf )
        # launch view
        viewRuns.run( idDataset, idConf, datasetName, runs )

class StartRunDoer( Doer ):

    def __init__( self, conn, configMachinesForms ) :
        super().__init__( conn )
        self.confMachinesForms = configMachinesForms

    def start( self, fenetre, idConfig ) :
        # Get form fields for machine
        config = db.getConfig( self.conn, idConfig )

        # Get machine name
        machineName = db.getMachineNameById( self.conn, config[ "idMachine" ] )
        # Get fields
        machineFields = self.confMachinesForms[ machineName ]

        # Launch run dialog
        startTrainingDialog = view.StartTrainDialog( fenetre, fenetre.doRunTraining, machineFields )
        startTrainingDialog.run()

class StartPredictDoer( Doer ):

    def __init__( self, conn ) :
        super().__init__( conn )

    def start( self, fenetre, idConfig ) :

        # Launch predict dialog
        startPredictDialog = view.StartPredictDialog( fenetre, fenetre.doRunPredict )
        startPredictDialog.run()
