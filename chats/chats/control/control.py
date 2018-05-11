import view.view as view
import db.db as db

class ConfigDoer:

    def __init__( self, conn ) :
        self.conn = conn

    def showHyperParams( self, idConf ):

        # Get config
        config = db.getConfig( self.conn, idConf )
        # get hyper parameters id
        idHyperParams = config[ "idHyperParams" ]
        # get hyper parameters
        hyperParams = db.getHyperParams( self.conn, idHyperParams )

        # Launch window, it may update hps
        viewHp = view.ViewOrUpdateHyperParamsUpdateWindow()
        
        newHyperParams = viewHp.run( hyperParams )
        print( "Result:", newHyperParams )
        
        if ( newHyperParams == None ) :
            ## Nothing to do
            return

        # Get or create new hyper parameters
        idNewHyperParams = db.getOrCreateHyperParams( self.conn, newHyperParams[ "hyper_params" ] )

        # check for change
        if ( idHyperParams != idNewHyperParams ) :
            # Update config
            updateConfig( self.conn, newHyperParams )
            #commit
            conn.commit()

    def newConfig( this ):
        print( "newConfig" )

    def updateConfig( this, idConf ):
        print( "updateConfig" )

    def deleteConfig( this, idConf ):
        print( "deleteConfig" )

