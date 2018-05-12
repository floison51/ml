import view.view as view
import db.db as db

class ConfigDoer:

    def __init__( self, conn ) :
        self.conn = conn

    def showHyperParams( self, fenetre, idConf ):

        # Get config
        self.config = db.getConfig( self.conn, idConf )
        # get hyper parameters id
        self.idHyperParams = self.config[ "idHyperParams" ]
        # get hyper parameters
        hyperParams = db.getHyperParams( self.conn, self.idHyperParams )

        # Launch window, it may update hps
        viewHp = view.ViewOrUpdateHyperParamsUpdateWindow( fenetre, self.updateHyperParams )
        
        # launch view with callback
        viewHp.run( hyperParams )

    def updateHyperParams( self, newHyperParams ):

        print( "Result:", newHyperParams )
        
        if ( newHyperParams == None ) :
            ## Nothing to do
            return

        # Get or create new hyper parameters
        idNewHyperParams = db.getOrCreateHyperParams( self.conn, newHyperParams )

        # check for change
        if ( self.idHyperParams != idNewHyperParams ) :
            # Update config
            self.config[ "idHyperParams" ] = idNewHyperParams
            db.updateConfig( self.conn, self.config )
            #commit
            self.conn.commit()
        
    def newConfig( self ):
        print( "newConfig" )

    def updateConfig( self, idConf ):
        print( "updateConfig" )

    def deleteConfig( self, idConf ):
        print( "deleteConfig" )

