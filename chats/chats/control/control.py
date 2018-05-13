import view.view as view
import db.db as db
import const.constants as const

class Doer:
    def __init__( self, conn ) :
        self.conn = conn

    
class HyperParamsDoer( Doer ):

    def __init__( self, conn ) :
        super().__init__( conn )

    def updateHyperParams( self, fenetre, idConf ):

        # Get config
        self.config = db.getConfig( self.conn, idConf )
        # get hyper parameters id
        self.idHyperParams = self.config[ "idHyperParams" ]
        # get hyper parameters
        hyperParams = db.getHyperParams( self.conn, self.idHyperParams )

        # get best hyper params
        ( bestHyperParams, bestDevAccuracy ) = db.getBestHyperParams( self.conn, idConf )
         
        # Launch window, it may update hps
        viewHp = view.ViewOrUpdateHyperParamsWindow( fenetre, self.doCreateOrUpdateHyperParams )
        
        # launch view with callback
        viewHp.run( hyperParams, bestHyperParams, bestDevAccuracy )

    def doCreateOrUpdateHyperParams( self, fenetre, newHyperParams ):

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
        
class ConfigDoer( Doer ):

    def __init__( self, conn ) :
        super().__init__( conn )

    def newConfig( self, fenetre ):
        "Create new config"
        viewConfig = view.CreateOrUpdateConfigWindow( fenetre, self.doCreateOrUpdateConfig )
        # launch view with callback
        viewConfig.run()

    def updateConfig( self, fenetre, idConf ):
        print( "updateConfig" )

    def deleteConfig( self, fenetre, idConf ):
        print( "deleteConfig", idConf )
        db.deleteConfig( self.conn, idConf )
        
        # Update window
        fenetre.deleteConfigGrid( idConf )
        # commit
        self.conn.commit()
        
    def doCreateOrUpdateConfig( self, fenetre, newConfig ):

        print( "Result:", newConfig )
        
        if ( newConfig == None ) :
            ## Nothing to do
            return
        
        # Default hyper params
        if ( not const.KEY_DICO_HYPER_PARAMS in newConfig ) :
            hyperParams = {}
            for ( key, hpCarac ) in const.HyperParamsDico.CARAC.items() :
                hyperParams[ key ] = hpCarac[ 1 ]                
            
        # Get or create new config
        idNewConfig = db.getOrCreateConfig( self.conn, newConfig[ "name" ], newConfig[ "structure" ], hyperParams )
        
        # commit
        self.conn.commit()        