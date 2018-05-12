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
    
    carac = {
        
        KEY_MINIBATCH_SIZE      : [ "int" ],
        KEY_NUM_EPOCHS          : [ "int" ],
        KEY_USE_WEIGHTS         : [ "boolean" ],
        KEY_START_LEARNING_RATE : [ "float" ],
        KEY_BETA                : [ "float" ],
        KEY_KEEP_PROB           : [ "float" ],
    } 
