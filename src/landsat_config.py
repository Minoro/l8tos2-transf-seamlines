CUDA_DEVICE = 0

EPOCHS = 50
BATCH_SIZE = 8
LR = 0.001
# Index of the bands to be read from the tif image
BANDS = (7,6,5,)

IMAGE_SHAPE = (256, 256, len(BANDS))

# Max. pixel value, used to normalize the Landsat-8 images
QUANTIFICATION_VALUE = 65535

MODEL = 'SegFormerB0'

# Mask name used to train the model
MASK = 'Voting'

IMAGES_DATAFRAMES_PATH = '../../resources/landsat/dataframes/'

# Path to Landsat-8 images
IMAGES_PATH = '../../landsat/dataset/images/patches/'
# Path to Landsat-8 active fire masks 
MASKS_PATH = '../../landsat/dataset/masks/voting/'

# Path to save the final models
LANDSAT_OUTPUT_DIR = '../../resources/landsat/output/'

EARLY_STOP_PATIENCE = 5 
CHECKPOINT_PERIOD = 'epoch'
CHECKPOINT_MODEL_NAME = 'checkpoint-epoch_{}_{}_{{epoch:02d}}.weights.h5'

FINAL_WEIGHTS_OUTPUT = 'model_{}_{}_{}_final_weights.h5'

LANDSAT_MANUAL_ANNOTATIONS_DATAFRAME_PATH = '../../resources/landsat/dataframes/'
LANDSAT_MANUAL_ANNOTATIONS_IMAGES_PATH = '../../resources/landsat/dataset/manual_annotations/patches/landsat_patches'
LANDSAT_MANUAL_ANNOTATIONS_MASK_PATH = '../../resources/landsat/dataset/manual_annotations/patches/manual_annotations_patches' 

