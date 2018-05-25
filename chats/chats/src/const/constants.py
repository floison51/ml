'''
Created on 28 avr. 2018
Machine Learning pseudo-constants
@author: fran
'''
# Dico keys
KEY_DICO_HYPER_PARAMS   = "hyperParameters"
KEY_DICO_SYSTEM         = "system"
KEY_DICO_PERF           = "perf"
KEY_DICO_DATA           = "data"
KEY_DICO_RESULT         = "result"

# hyper-parameters keys
KEY_MINIBATCH_SIZE      = "minibatch_size"
KEY_NUM_EPOCHS          = "num_epochs"
KEY_USE_WEIGHTS         = "isUseWeights"
KEY_START_LEARNING_RATE = "start_learning_rate"
KEY_BETA                = "beta"
KEY_KEEP_PROB           = "keepProb"
KEY_DECAY_NB            = "decayNb"
KEY_DECAY_PERCENT       = "decayPercent"

# system info keys
KEY_HOSTNAME            = "hostname"
KEY_TENSOR_FLOW_VERSION = "tensorFlowVersion"
KEY_OS_NAME             = "osName"

# data kets
KEY_TRN_SIZE    = "trnSize"
KEY_DEV_SIZE    = "devSize"
KEY_TRN_SHAPE   = "trainigShape"
KEY_DEV_SHAPE   = "devShape"
KEY_DEV_Y_SIZE  = "yDevSize"
KEY_DEV_Y_SHAPE = "yDevShape"
KEY_TRN_Y_SIZE  = "yTrnSize"
KEY_TRN_Y_SHAPE = "yTrnShape"

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
        KEY_MINIBATCH_SIZE      : [ "int"       , 64        ],
        KEY_NUM_EPOCHS          : [ "int"       , 2000      ],
        KEY_USE_WEIGHTS         : [ "boolean"   , False     ],
        KEY_START_LEARNING_RATE : [ "float"     , 0.0001    ],
        KEY_BETA                : [ "float"     , 0.        ],
        KEY_KEEP_PROB           : [ "float"     , 1.        ],
        KEY_DECAY_NB            : [ "int"       , 10000     ],
        KEY_DECAY_PERCENT       : [ "float"     , 0.96      ],
    }

class ConfigsDico:

    KEYS = { "id", "idMachine", "idHyperParams" }
    CARAC = {
        # Type, default value
        "id"           : [ "int" , None ],
        "name"         : [ "string" , None ],
        "structure"    : [ "string" , None ],
        "machine"      : [ "string" , None ],
        "bestAccuracy" : [ "float" , None ],
    }
    OBJECT_FIELDS  = [ "id", "name", "structure", "idMachine", "idHyperParams" ]
    DISPLAY_FIELDS = [ "id", "name", "structure", "machine", "bestAccuracy" ]

class MachinesDico:

    KEYS = { "id" }
    CARAC = {
        # Type, default value
        "name"      : [ "string" , "None" ],
        "class"     : [ "string" , "learning.NoneMachine" ],
    }

class RunsDico:

    KEYS = { "id", "idConf", "idHyperParams" }
    CARAC = {
        # Type, default value
        "id"                : [ "int" , None ],
        "date"              : [ "datetime" , None ],
        "comment"           : [ "string" , None ],
        "perf_index"        : [ "float" , None ],
        "elapsed_second"    : [ "int" , None ],
        "train_accuracy"    : [ "float" , None ],
        "dev_accuracy"      : [ "float" , None ],
        "json_system_info"  : [ "json" , None ],
        "json_data_info"    : [ "json" , None ],
        "json_perf_info"    : [ "json" , None ],
        "json_result_info"  : [ "json" , None ],
        
    }
    OBJECT_FIELDS  = [ "id", "date", "comment", "perf_index", "elapsed_second", "train_accuracy", "dev_accuracy", "json_system_info", "json_data_info", "json_perf_info", "json_result_info" ]
    DISPLAY_FIELDS = OBJECT_FIELDS
