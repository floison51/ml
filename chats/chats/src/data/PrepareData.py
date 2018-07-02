#!/usr/local/bin/python2.7
# encoding: utf-8
'''
data.PrepareData -- shortdesc

data.PrepareData is a description

It defines classes_and_methods

@author:     user_name

@copyright:  2018 organization_name. All rights reserved.

@license:    license

@contact:    user_email
@deffield    updated: Updated
'''

import sys
import os

import glob
import random
from PIL import Image
import h5py
import numpy as np

#import wget
import shutil

#from apiclient.discovery import build

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

__all__ = []
__version__ = 0.1
__date__ = '2018-04-24'
__updated__ = '2018-04-24'

DEBUG = 1
TESTRUN = 0
PROFILE = 0

# Constants
TRAINING_TEST_SET_PC = 0.90
# Small traning set to rapid iterations
#TRAINING_TEST_SET_PC = 0.10

class CLIError(Exception):
    '''Generic exception to raise and log different fatal errors.'''
    def __init__(self, msg):
        super(CLIError).__init__(type(self))
        self.msg = "E: %s" % msg
    def __str__(self):
        return self.msg
    def __unicode__(self):
        return self.msg

def deleteDirContent( dir ) :
    # Delete files in dir
    for the_file in os.listdir( dir ):
        file_path = os.path.join( dir, the_file )
        try:
            if os.path.isfile( file_path ):
                os.unlink( file_path )
            elif os.path.isdir( file_path ):
                shutil.rmtree( file_path )
        except Exception as e:
            print(e)

    # check deleted
    if ( len( os.listdir( dir ) ) != 0 ) :
        print( "Dir", dir, "not empty." )
        sys.exit( 1 )

def transformImages( fromDir, toDir, files, what ):

    for oriImgFile in files :
        # Copy from image
        # remove fromDir prefix
        toFilePrefix = oriImgFile[ len( fromDir ) + 1 : ]
        toFile = toDir + "/" + what + "/" + toFilePrefix

        ## Make target dir
        toFileDir = os.path.dirname( toFile )
        if ( not os.path.exists( toFileDir ) ) :
            os.makedirs( toFileDir, exist_ok = True )

        ## Load image
        img = Image.open( oriImgFile )

        if ( what == "original" ) :

            # Copy ori file
            shutil.copyfile( oriImgFile, toFile )

        elif ( what == "flip" ) :

            ## Flip verticaly image
            flippedImage = img.transpose( Image.FLIP_LEFT_RIGHT )

            ## save it
            toFlippedImage = toDir + "/" + what + "/" + toFilePrefix + "-flipped.png"
            flippedImage.save( toFlippedImage, 'png' )

        elif ( what == "rotate" ) :

            # +15 degrees
            rotPlusImage = img.rotate( +15, expand=True )
            toRotPlusImage = toDir + "/" + what + "/" + toFilePrefix + "-rotP15.png"
            rotPlusImage.save( toRotPlusImage, 'png' )

            # -15 degrees
            rotMinusImage = img.rotate( -15, expand=True )
            toRotMinusImage = toDir + "/" + what + "/" + toFilePrefix + "-rotM15.png"
            rotMinusImage.save( toRotMinusImage, 'png' )

        else :
            raise ValueError( "Unknown image operation '" + what + "'" )

# Create dev and test sets

def buildDataSet( dataDir, what, baseDir, files, iStart, iEnd, size, outFileName, indexLabel = 0 ):
    # images list
    imagesList = []
# y : cat or non-cat
    y = []
# tags of images
    tags = []
# pathes of images
    pathes = []

# From first image to dev test set percentage of full list
#for i in range( iEndTestSet ):
    for i in range( iStart, iEnd ):

        curImage = files[i]
        # get rid of basedir
        relCurImage = curImage[ len( baseDir ) + 1: ]
        relCurImage = relCurImage.replace( '\\', '/' )

        relCurImageSegments = relCurImage.split( "/" )

        # Cat?
        isCat = relCurImageSegments[ indexLabel ] == "cats"

        # tags
        _tag = relCurImageSegments[ indexLabel + 1 ]

        # load image
        img = Image.open( curImage )
        # Resize image
        resizedImg = img.resize( ( size, size ) )
        # populate lists
        pix = np.array(resizedImg)
        # pix.spahe = (64, 64, 3 ) pour éviter les images monochromes
        if (pix.shape != ( size, size, 3 ) ):
            print("Skipping image", curImage)
            print("It's not a (NxNx3) image.")
        else:
            imagesList.append( pix )
            y.append( isCat )                       # Image will be saved
            tags.append( np.string_( _tag ) )       # Label of image

            # Image rel path from home
            relHomeCurImage = curImage[ len( dataDir ) + 1: ]
            relHomeCurImage = relHomeCurImage.replace( '\\', '/' )
            pathes.append( np.string_( relHomeCurImage ) )

# Store as binary stuff

    # Check dims
    for i in range( len( imagesList ) ) :
        image = imagesList[ i ]
        imageShape = image.shape
        if ( imageShape != ( size, size, 3 ) ) :
            print( "Wrong image:", pathes[ i ] )
            sys.exit( 1 )

    preparedDir = dataDir + "/prepared/" + what
    os.makedirs( preparedDir, exist_ok = True )
    absOutFile = preparedDir + "/" + outFileName

    # remove data dir
    relImgDir = baseDir[ len( dataDir ) + 1 : ]

    with h5py.File( absOutFile, "w") as dataset:
        dataset["x"]        = imagesList
        dataset["y"]        = y
        dataset["tags"]     = tags
        dataset["pathes"]   = pathes
        dataset["imgDir"]   = relImgDir.encode( 'utf-8' )

def createTrainAndDevSets( name, transformations, pc ):

    # current dir for data
    dataDir = os.getcwd().replace( "\\", "/" )
    dataDir += "/" + name

    # Clean target dirs
    transformedDir = dataDir + "/transformed"
    os.makedirs( transformedDir, exist_ok = True )
    deleteDirContent( transformedDir )

    preparedDir = dataDir + "/prepared"
    os.makedirs( preparedDir, exist_ok = True )
    deleteDirContent( preparedDir )

    # Base dir for cats and not cats images
    oriDir = dataDir + "/images"

    # get original images
    oriFiles = glob.glob( oriDir + '/**/*.*', recursive=True)

    # Shuffle files
    random.shuffle( oriFiles )

    # for debug : only 20 images
    # oriFiles = oriFiles[ 0:20 ]

    sizes = ( 64, 92, 128 )

    print( "Build DEV data set" )
    iEndTrainingSet = int( len( oriFiles ) * pc );

    for size in sizes :
        ## Build DEV data set
        buildDataSet( \
            dataDir, "dev", oriDir, oriFiles, \
            iEndTrainingSet, len( oriFiles ), \
            size, "dev_chats-" + str( size) + ".h5" \
        )

    ## transform images for DEV
    transformedDir = dataDir + "/transformed"

    # Original files for TRN data set
    trnOriFiles = oriFiles[ 0 : iEndTrainingSet ]

    # transformations
    for transformation in transformations :

        transformImages( oriDir, transformedDir, trnOriFiles, transformation )

        targetFiles = glob.glob( transformedDir + '/' + transformation + '/**/*.*', recursive=True)
        buildTrnSet( dataDir, transformedDir, targetFiles, transformation, sizes )

def buildTrnSet( dataDir, transformedDir, targetFiles, what, sizes ):

    # Shuffle files
    random.shuffle( targetFiles )

    for size in sizes :
        print( "Build TRAINING data set '" + what + "' - size", size, "length:", len( targetFiles ) )

        ## Build TRN data set from transformed dir
        buildDataSet( \
            dataDir, "trn/" + what, transformedDir, targetFiles, \
            0, len( targetFiles ), \
            size, "train_chats-" + str( size) + ".h5", \
            indexLabel = 1 # rel path is original/cats/label/xxx.jpg, so label is in index 1
        )

    print( "Finished" );


def main(argv=None): # IGNORE:C0111
    '''Command line options.'''

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    program_name = os.path.basename(sys.argv[0])
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
    program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
    program_license = '''%s

  Created by user_name on %s.
  Copyright 2018 organization_name. All rights reserved.

  Licensed under the Apache License 2.0
  http://www.apache.org/licenses/LICENSE-2.0

  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.

USAGE
''' % (program_shortdesc, str(__date__))

    try:
        # Setup argument parser
        parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)
        parser.add_argument('-V', '--version', action='version', version=program_version_message)

        parser.add_argument( "-command" );

        # Process arguments
        args = parser.parse_args()

        # check actions
        command = args.command
#         if ( "googleImage".equals( command ) ) :
#             print( "XXX" )
#             # get data
#             getGoogleImageData()


    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        return 0
    except Exception as e:
        if DEBUG or TESTRUN:
            raise( e )
        indent = len(program_name) * " "
        sys.stderr.write(program_name + ": " + repr(e) + "\n")
        sys.stderr.write(indent + "  for help use --help")
        return 2

if __name__ == "__main__":

    # Make sure random is repeatable
    random.seed( 1 )

    input( "Type enter to continue" )

    createTrainAndDevSets( "hand-made", ( "original", "flip", "rotate", ), TRAINING_TEST_SET_PC )
    createTrainAndDevSets( "contest", ( "original", ), 100 )
