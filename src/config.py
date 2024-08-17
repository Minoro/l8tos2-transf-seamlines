################################ GERAL #############################################################

# Caminho para os patches anotados manualmente
MANUAL_ANNOTATIONS_IMAGE_PATH = '../../resources/sentinel/Sentinel2/manual_annotated/gdal_croped/imgs_256'
# Caminho para as máscaras da anotação manual
MANUAL_ANNOTATIONS_MASK_PATH = '../../resources/sentinel/Sentinel2/manual_annotated/gdal_croped/annotations'
# Caminho para as predições de cada algoritmo de thresholding
METHODS_PREDICTIONS_ANNOTATED_PATH = '../../resources/sentinel/Sentinel2/manual_annotated/gdal_croped/methods'

METHODS_PREDICTIONS_SEAMLINE_PATH = '../../resources/seamlines/gdal_algorithms_441'

# Caminho para as imagens com costura
SEAMLINE_IMAGES_PATH = '../../resources/seamlines/gdal_patches_441'

# Formato do patch de imagem para as redes
IMAGE_SHAPE = (256,256,3)

# Formato da máscara final
MASK_SHAPE = (256,256)

# Marcador no nome dos patches para identificar as imagens anotadas manualmente
ANNOTATION_IMAGE_MARKER = '_20m_stack_'

# Marcador no nome dos patches para identificar as imagens de costura
SEAMLINE_IMAGE_MARKER = '_b12b11b8a_'

# Marcador no nome dos patches anotados manualmente
ANNOTATION_MASK_MARKER = '_20m_stack_maskf_'

# Marcador no nome das predições geradas pelos métodos de limiares
METHOD_PREDICTION_MARKER = '_mask_'

# Algoritmos de limiares disponíveis
THRESHOLDING_METHODS = [
    'KatoNakamura', 'Liu', 'Murphy'
]

RANDOM_SEED = 42

######################### TRANSFER LEARNING #####################################################

# GPU para ser utilizada
CUDA_DEVICE = 0

# Identificação para separar os experimentos.
IDENTIFICATION_PREFIX = ''

IDENTIFICATION_TRAIN_PREFIX = 'scratch'

# Modelo base para aplicar o transfer learning
# Disponível: unet, deeplabv3+ e SegFormerB0
MODEL = 'unet'

# Modelo base treinado no Landsat-8 (Apenas Voting está disponível)
BASE_MODEL = 'Voting'

# Estratégia de Transfer Learning (Apenas unfreeze está disponível)
TRANSFER_LEARNING_STRATEGY = 'unfreeze'

# Valor para dividir os pixels das imagens do Sentinel-2 (uint16)
SENTINEL_QUANTIFICATION_VALUE = 10000.0

# Modo de normalização das imagens do Sentinel-2 (Disponível: 'bn' (adiciona BN no inicio e performa normalização) 'no-bn' (sem BN no inicio, mas performa normalização) None (sem camada de normalização) )
NORMALIZATION_MODE = 'no-bn'

# Se verdadeiro performa Fine Tuning (desde que o número de epocas seja maior que zero)
FINE_TUNING = True

# Tamanho do batch para Transfer Learning
BATCH_SIZE = 8

# Learning rate para o transfer learning
LR = 1e-4

# Número de épocas para transfer learning. Se zero não performa transfer learning
EPOCHS = 40

# Número de épocas para Early Stopping
EARLY_STOP_PATIENCE = 5

# Se verdadeiro recarrega os melhores pesos com base no loss
EARLY_STOP_RESTORE_BEST = True

# Loss function do modelo
LOSS_FUNCTION = 'bce'

# Se verdadeiro aplica flip horizontal e vertical sobre as imagens de TREINO
USE_DATA_AUGMENTATION = True

PRETRAINED_WEIGHTS_PATH = '../../resources/landsat/weights'

PRETRAINED_MODELS = ['unet', 'deeplabv3+', 'SegFormerB0']

OUTPUT_RESULTS_TRANSFER_LEARNING_PATH = '../../resources/output/results/transfer_learning'

OUTPUT_WEIGHTS_TRANSFER_LEARNING_PATH = '../../resources/output/weights/transfer_learning'

##################### TREINO DO ZERO ######################################################

OUTPUT_RESULTS_TRAIN_FROM_SCRATCH_PATH = '../../resources/output/results/scratch'

OUTPUT_WEIGHTS_TRAIN_FROM_SCRATCH_PATH = '../../resources/output/weights/transfer_learning'


CSV_FROM_SCRATCH_PARTIAL_DATASET_BASE_PATH = '../../resources/sentinel/dataframe_partial_dataset/'

OUTPUT_RESULTS_TRAIN_FROM_SCRATCH_PARTIAL_DATASET_PATH = '../../resources/output/results/scratch_partial_dataset'

OUTPUT_WEIGHTS_TRAIN_FROM_SCRATCH_PARTIAL_DATASET_PATH = '../../resources/output/weights/transfer_learning'

######################### PREDIÇÕES ############################################################

# Pasta onde serão salvas as predições das redes após o transfer learning
OUTPUT_PREDICTIONS_TRANSFER_LEARNING_PATH = '../../resources/output/predictions/transfer_learning'


######################### AVALIAÇÃO DAS MASCARAS ##########################################################

# Se verdadeiro precarrega as anotações manuais para memória e utiliza como cache
PRELOAD_ANNOTATIONS_TO_COMPARE_WITH_METHODS_PREDICTIONS = True

# Se verdadeiro carrega todas as predições do método para memória e utiliza como cache 
PRELOAD_METHODS_PREDICTIONS = False

# Caminho em que os resultados dos algoritmos serão salvos
OUTPUT_RESULTS_METHODS_PREDICTIONS_PATH = '../../resources/output/results/thresholding_methods'

######################## FOLDS #############################################################################

# CSV com informação de número de pixel de fogo para cada patch. 
# Utilizado para gerar os folds.
# Se o arquivo não existir, é gerado ao rodar o script de gerar folds
CSV_NUM_FIRE_PIXELS_PER_PATCH_PATH = '../../resources/sentinel/kfold/statistics/num_fire_pixels_per_patch.csv'

# Se verdadeiro, computa novamente o número de pixels de fogo das imagens antes de gerar os folds
OVERRIDE_CSV_NUM_FIRE_PIXELS_PER_PATCH = False

# Caminho base para o diretório com os folds. Dentro desta pasta haverá arquivos em csv com a divisão dos folds
CSV_FOLDS_BASE_PATH = '../../resources/sentinel/kfold/'

# Número total de folds
NUM_FOLDS = 5

# Gera os folds estratificando por "categoria" (patches com fogo, patches sem fogo anotado, patches de costura - sem fogo anotado)
STRATIFIED_FOLDS = False

# Se verdadeiro gera reserva um split de tamanho igual ao split de teste para validação
# O split é escolhido do cojunto de treino
GENERATE_VALIDATION_FOLD = True

