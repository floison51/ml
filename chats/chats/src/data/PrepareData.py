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

import tensorflow as tf

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

def buildDataSet( dataDir, what, baseDir, files, iStart, iEnd, size, outFileNameNoExt, indexLabel = 0 ):
    # images list
    imagesNumpyList = []

# y : cat or non-cat
    yNumpyList = []
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
        resizedPix = np.array( resizedImg )
        # pix.shape = (64, 64, 3 ) pour éviter les images monochromes
        if ( resizedPix.shape != ( size, size, 3 ) ):
            print("Skipping image", curImage)
            print("It's not a (NxNx3) image.")
        else:

            imagesNumpyList.append( resizedPix )

            yNumpyList.append( isCat )              # Image will be saved

            tags.append( np.string_( _tag ) )       # Label of image

            # Image rel path from home
            relHomeCurImage = curImage[ len( dataDir ) + 1: ]
            relHomeCurImage = relHomeCurImage.replace( '\\', '/' )
            pathes.append( np.string_( relHomeCurImage ) )

# Store as binary stuff

    # Check dims
    for i in range( len( imagesNumpyList ) ) :
        imageNumpy = imagesNumpyList[ i ]
        imageShape = imageNumpy.shape
        if ( imageShape != ( size, size, 3 ) ) :
            print( "Wrong image:", pathes[ i ] )
            sys.exit( 1 )

    preparedDir = dataDir + "/prepared/" + what
    os.makedirs( preparedDir, exist_ok = True )
    absOutFileNoExt = preparedDir + "/" + outFileNameNoExt

    # Convert to numpy array
    tags_np   = np.array( tags ).T
    # passer de (476,) (risque) a  (476,1)
    tags_np    = tags_np.reshape( tags_np.shape[0], 1 )

    pathes_np = np.array( pathes ).T
    # passer de (476,) (risque) a  (476,1)
    pathes_np    = pathes_np.reshape( pathes_np.shape[0], 1 )

    # remove data dir
    relImgDir = baseDir[ len( dataDir ) + 1 : ]

    # Python h5 dataset
    createH5PYdataset( absOutFileNoExt, ( size, size, 3 ), ( 1, ), imagesNumpyList, yNumpyList, tags_np, pathes_np, relImgDir )

    # Tensorflow TFRecord format
    createTFRdataset( absOutFileNoExt, ( size, size, 3 ), ( 1, ), imagesNumpyList, yNumpyList, tags, pathes, relImgDir )

def createMetadata( absOutFileNoExt, tags, pathes, nbSamples, shape_X, shape_Y, relImgDir, dico ) :

    absOutFile = absOutFileNoExt + ".h5"

    with h5py.File( absOutFile, "w") as dataset:

        dataset[ "tags" ]     = tags
        dataset[ "pathes" ]   = pathes

        # Add attributes
        dataset.attrs[ "imgDir"]     = relImgDir.encode( 'utf-8' )
        dataset.attrs[ "nbSamples" ] = nbSamples
        dataset.attrs[ "shape_X"  ]  = shape_X
        dataset.attrs[ "shape_Y"  ]  = shape_Y

        # Add provided dico entries
        for key in dico :
            dataset[ key ] = dico[ key ]

def createH5PYdataset( absOutFileNoExt, shape_X, shape_Y, imagesNumpyList, yNumpyList, tags, pathes, relImgDir ) :

    nbSamples = len( imagesNumpyList )
    # assert same number of samples in X and Y
    assert( nbSamples == len( yNumpyList ) )

    createMetadata(
        absOutFileNoExt, tags, pathes,
        # Shapes
        nbSamples, shape_X, shape_Y,
        relImgDir,
        # Specific data
        { "X": imagesNumpyList, "Y": yNumpyList }
    )

def _floats_feature( values ):
    return tf.train.Feature( float_list = tf.train.FloatList( value=values ) )

def _int64_feature( value ):
    return tf.train.Feature(int64_list=tf.train.Int64List( value=[ value ] ))

def _ints64_feature( values ):
    return tf.train.Feature(int64_list=tf.train.Int64List( value=values ) )

def _bytes_feature( value ):
    return tf.train.Feature( bytes_list = tf.train.BytesList( value=[ value ] ) )

def createTFRdataset( absOutFileNoExt, shape_X, shape_Y, xNpList, yNpList, tags, pathes, relImgDir ) :

    # Meta-data
    absOutFileMetadata = absOutFileNoExt + "-tfrecord-metadata"
    absOutFile         = absOutFileNoExt + ".tfrecord"

    nbSamples = len( xNpList )
    # assert same number of samples in X and Y
    assert( nbSamples == len( yNpList ) )

    createMetadata(
        absOutFileMetadata, tags, pathes,
        # Shapes
        nbSamples, shape_X, shape_Y,
        relImgDir,
        {
            "XY_tfrecordPath": os.path.basename( absOutFile ).encode( 'utf-8' )
        }
    )

    # Create TFRecords file

    with tf.python_io.TFRecordWriter( absOutFile ) as tfRecordWriter:
        for i in range( len( xNpList ) ) :

            # Build TF train record for current image
            tfr = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'X': _bytes_feature( xNpList[ i ].reshape( -1 ).tobytes() ),
                        'Y': _int64_feature(  int( yNpList[ i ] ) )
                    }
                )
            )

            # write record
            tfRecordWriter.write( tfr.SerializeToString() )


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
    #oriFiles = oriFiles[ 0:20 ]

    sizes = ( 64, 92, 128 )

    print( "Build DEV data set" )
    iEndTrainingSet = int( len( oriFiles ) * pc );

    for size in sizes :
        ## Build DEV data set
        buildDataSet( \
            dataDir, "dev", oriDir, oriFiles, \
            iEndTrainingSet, len( oriFiles ), \
            size, "dev_chats-" + str( size) \
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
            size, "train_chats-" + str( size), \
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

    # Transformation have to be debugged: 25% of data, not 100%
    #createTrainAndDevSets( "hand-made", ( "original", "flip", "rotate", ), TRAINING_TEST_SET_PC )
    createTrainAndDevSets( "hand-made", ( "original", ), TRAINING_TEST_SET_PC )

    createTrainAndDevSets( "contest", ( "original", ), 1 - 0.02 )
