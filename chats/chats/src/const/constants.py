'''
Created on 28 avr. 2018
Machine Learning pseudo-constants
@author: fran
'''
import os

APP_KEY = "chats"
APP_RUN_KEY = "chats"

DB_DIR = os.getcwd().replace( "\\", "/" ) + "/run/db/" + APP_RUN_KEY

# Dico keys
KEY_DICO_HYPER_PARAMS   = "hyperParameters"
KEY_DICO_SYSTEM         = "system"
KEY_DICO_PERF           = "perf"
KEY_DICO_DATA           = "data"
KEY_DICO_RESULT         = "result"
KEY_DICO_DATASET_NAME   = "datasetName"

# hyper-parameters keys
KEY_MINIBATCH_SIZE      = "minibatch_size"
KEY_NUM_EPOCHS          = "num_epochs"
KEY_USE_WEIGHTS         = "isUseWeights"
KEY_START_LEARNING_RATE = "start_learning_rate"
KEY_BETA                = "beta"
KEY_KEEP_PROB           = "keepProb"
KEY_LEARNING_RATE_DECAY_NB_EPOCH = "decayNb"
KEY_LEARNING_RATE_DECAY_PERCENT = "decayPercent"
KEY_USE_BATCH_NORMALIZATION     = "useBatchNormalization"

# system info keys
KEY_PYTHON_VERSION      = "pythonVersion"
KEY_HOSTNAME            = "hostname"
KEY_TENSOR_FLOW_VERSION = "tensorFlowVersion"
KEY_OS_NAME             = "osName"

# data kets
KEY_TRN_X_SIZE    = "xTrnSize"
KEY_TRN_X_SHAPE   = "xTrnShape"

KEY_DEV_X_SIZE    = "xDevSize"
KEY_DEV_X_SHAPE   = "xDevShape"

KEY_TRN_Y_SIZE    = "yTrnSize"
KEY_TRN_Y_SHAPE   = "yTrnShape"

KEY_DEV_Y_SIZE    = "yDevSize"
KEY_DEV_Y_SHAPE   = "yDevShape"

KEY_IS_SUPPORT_BATCH_STREAMING = "KEY_IS_SUPPORT_BATCH_STREAMING"

# result keys
KEY_TRN_NB_ERROR_BY_TAG = "trnNbErrorsByTag"
KEY_TRN_PC_ERROR_BY_TAG = "trnNbErrorsPcByTag"
KEY_DEV_NB_ERROR_BY_TAG = "devNbErrorsByTag"
KEY_DEV_PC_ERROR_BY_TAG = "devNbErrorsPcByTag"

# Performance keys
KEY_PERF_IS_USE_TENSORBOARD         = "isUseTensorboard"
KEY_PERF_IS_USE_FULL_TENSORBOARD    = "isUseFullTensorboard"

class HyperParamsDico:

    KEYS = { id }
    CARAC = {
        # Type, default value
        KEY_MINIBATCH_SIZE      : [ "int"       , 64     , None            ],
        KEY_NUM_EPOCHS          : [ "int"       , 2000   , None            ],
        KEY_USE_WEIGHTS         : [ "boolean"   , False  , None            ],
        KEY_START_LEARNING_RATE : [ "float"     , 0.0001 , "{0:.0000000f}" ],
        KEY_BETA                : [ "float"     , 0.     , "{0:.0000000f}" ],
        KEY_KEEP_PROB           : [ "float"     , 1.     , "{0:.00f}"      ],
        
        KEY_LEARNING_RATE_DECAY_NB_EPOCH  : [ "int"       , 500    , None            ],
        KEY_LEARNING_RATE_DECAY_PERCENT   : [ "float"     , 0.96   , "{0:.00f}%"     ],
        
        KEY_USE_BATCH_NORMALIZATION : [ "boolean", False, None ]
    }

class DatasetDico:

    KEYS = { "id" }
    CARAC = {
        # Type, default value
        "id"           : [ "int"    , None, None ],
        "name"         : [ "string" , None, None ],
        "displayOrder" : [ "string" , None, None ],
        "pathHome"      : [ "string" , None, None ],
        "pathTrn"      : [ "string" , None, None ],
        "pathDev"      : [ "string" , None, None ],
    }
    OBJECT_FIELDS  = [ "id", "name", "displayOrder", "pathHome", "pathTrn", "pathDev" ]
    DISPLAY_FIELDS = OBJECT_FIELDS

class ConfigsDico:

    KEYS = { "id", "idMachine" }
    CARAC = {
        # Type, default value
        "id"              : [ "int"    , None, None ],
        "name"            : [ "string" , None, None ],
        "machine"         : [ "string" , None, None ],
        "imageSize"       : [ "int"    , 64  , None ],
        "structure"       : [ "string" , None, None ],
        "bestDevAccuracy" : [ "float" ,  None, "{:.2f}" ],
        "assoTrnAccuracy" : [ "float" ,  None, "{:.2f}" ],
    }
    OBJECT_FIELDS  = [ "id", "name", "structure", "imageSize", "idMachine" ]
    DISPLAY_FIELDS = [ "id", "name", "machine", "imageSize", "structure", "bestDevAccuracy", "assoTrnAccuracy" ]

class MachinesDico:

    KEYS = { "id" }
    CARAC = {
        # Type, default value
        "name"      : [ "string" , "None"                , None ],
        "class"     : [ "string" , "learning.NoneMachine", None ],
    }

class RunsDico:

    KEYS = { "id", "idDataset", "idConf", "idHyperParams" }
    CARAC = {
        # Type, default value
        "id"                    : [ "int"      , None, None     ],
        "idDataset"             : [ "int"      , None, None     ],
        "idConf"                : [ "int"      , None, None     ],
        "idHyperParams"         : [ "int"      , None, None     ],
        "dateTime"              : [ "datetime" , None, None     ],
        "comment"               : [ "string"   , None, None     ],
        "perf_index"            : [ "float"    , None, "{:.1f}" ],
        "elapsed_second"        : [ "int"      , None, None ],
        "train_accuracy"        : [ "float"    , None, "{:.2f}" ],
        "dev_accuracy"          : [ "float"    , None, "{:.2f}" ],
        "json_conf_saved_info"  : [ "json"     , None, None     ],
        "json_system_info"      : [ "json"     , None, None     ],
        "json_data_info"        : [ "json"     , None, None     ],
        "json_perf_info"        : [ "json"     , None, None     ],
        "json_result_info"      : [ "json"     , None, None     ],
        
    }
    OBJECT_FIELDS  = [ "id", "dateTime", "comment", "perf_index", "elapsed_second", "train_accuracy", "dev_accuracy", "json_conf_saved_info", "json_system_info", "json_data_info", "json_perf_info", "json_result_info" ]
    DISPLAY_FIELDS = [ "comment", "idDataset", "idConf", "idHyperParams", "train_accuracy", "dev_accuracy", "perf_index", "elapsed_second", "dateTime", "conf_saved_info", "system_info", "data_info", "perf_info", "result_info" ]

