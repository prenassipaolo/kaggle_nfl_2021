# default paths of configuration file and model weights file
PATH = "deepsort/"
FILNAME = 'config.yaml'
WEIGHTS = 'ckpt.t7'

FILE_PATH = PATH + FILNAME

# parameters
REID_CKPT = PATH + WEIGHTS
MAX_DIST = 0.2
MIN_CONFIDENCE = 0.3
NMS_MAX_OVERLAP = 0.5
MAX_IOU_DISTANCE = 0.9
MAX_AGE = 14
N_INIT = 1
NN_BUDGET = 30

STRING_PARAMETERS = f'''\
DEEPSORT:
    REID_CKPT: "{REID_CKPT}"
    MAX_DIST: {MAX_DIST}
    MIN_CONFIDENCE: {MIN_CONFIDENCE}
    NMS_MAX_OVERLAP: {NMS_MAX_OVERLAP}
    MAX_IOU_DISTANCE: {MAX_IOU_DISTANCE}
    MAX_AGE: {MAX_AGE}
    N_INIT: {N_INIT}
    NN_BUDGET: {NN_BUDGET}
'''

def create_config(file_path = FILE_PATH, string_parameters = STRING_PARAMETERS):
    """
    Creates yaml file with all the parameters for the deepsort algorithm
    """


    

    with open(file_path, mode='w') as file:
        file.write(string_parameters)
