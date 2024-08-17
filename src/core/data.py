import tensorflow as tf
import rasterio
import numpy as np

RANDOM_SEED = 42

SEAMLINE_MASK = np.zeros((256,256,1))
NOFIRE_MASK = np.zeros((256,256,1))

g = None


def open_image_and_mask(paths):
    """Abre uma imagem e a máscara a partir do tensor contendo os caminhos
    A primeira posição do tensor deve conter o caminho para a imagem
    A segunda posição do tensor deve conter o caminho para a máscra
    """
    # print(paths)
    image_path = paths[0].numpy().decode()
    mask_path = paths[1].numpy().decode()

    # Lê as bandas SWIR-2, SWIR-1 e NIR, nessa ordem
    # O transpose é aplicado para converter para channels-last
    with rasterio.open(image_path) as src:
        if src.meta['count'] == 3:
            # Imagem de costura com as bandas b12 b11 b8a (nessa ordem)
            img = src.read().transpose((1,2,0))
        else:
            # Imagem da stack de 20m
            img = src.read((6,5,4)).transpose((1,2,0))

    # Lê a máscara com a dimensão do canal (256,256,1)
    if mask_path == '':
        return img, SEAMLINE_MASK
    
    with rasterio.open(mask_path) as src:
        mask = src.read().transpose((1,2,0))
    
    return img, mask


def get_dataset_from_paths(image_paths, masks_paths, batch_size=8, normalization_layer=None, use_data_augmentation=False, shuffle=False, repeat=True):
    """ Returns a tf.data.Dataset to feed the networks.
    It can be used to iterate over the dataset, it generates the batch of images and masks.
    Returns a tensor with the shape [[BATCH_SIZE, 256, 256, 3], [BATCH_SIZE, 256, 256, 1]].
    """

    # Adjust the paths to be used by the dataset
    dataset = tf.data.Dataset.from_tensor_slices([*zip(image_paths, masks_paths)])
    if shuffle:
        # Shuffle the paths instead of the "open" images to save memory
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=RANDOM_SEED) 

    # Give the paths of the images and masks to the function "open_image_and_mask" which return the tensor representing the images and masks
    dataset = dataset.map(lambda x:  tf.py_function(open_image_and_mask, [x], [tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
    
    return prepare_dataset(dataset, batch_size, normalization_layer, use_data_augmentation, repeat)
    

def prepare_dataset(dataset, batch_size=8, normalization_layer=None, use_data_augmentation=False, repeat=True):
    """Apply the transformations over the tf.data.Dataset.
    It returns dataset adjusted to generate batchs.
    The normalization_layer will be used to transform the images to a new base scale
    The final dataset will be returned and can be used to iterate over the batchs of images and masks
    """

    if normalization_layer is not None:
        dataset = dataset.map(lambda x,y: (normalization_layer(x), y))
                
    # dataset = dataset.cache()
    if repeat:
        # repete o dataset para formar os batchs do mesmo tamanho
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size) 
    
    if use_data_augmentation:
        dataset = dataset.map(data_augmentation(RANDOM_SEED))
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset




def data_augmentation(random_seed=None):
    # Define o gerador a partir da semente para resultados mais consistentes
    g = tf.random.Generator.from_seed(random_seed)
    
    def augmentation(input_image, input_mask):
        rand = g.uniform([1])
        # tf.print('Rand 1:', rand)
        if rand > 0.5:
            # tf.print('Flip horizontal')
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)
            
        rand = g.uniform([1])
        # tf.print('Rand 2:', rand)
        if rand > 0.5:
            # tf.print('Flip vertical')
            input_image = tf.image.flip_up_down(input_image)
            input_mask = tf.image.flip_up_down(input_mask)

        return input_image, input_mask

    return augmentation