"""
Esse código remove os patches do Sentinel-2 que não possuem informação válida (totalmente preenchidos com zero).
O valor zero é reservado para informar "No-Data" no Sentinel, remover esses patches evita que sejam incluídos nos folds de treino.
"""

import os
import sys
import rasterio
import shutil
from glob import glob
from tqdm.auto import tqdm

MANUAL_ANNOTATIONS_IMAGE_PATH = '../../resources/sentinel/Sentinel2/manual_annotated/gdal_croped/imgs_256'
MANUAL_ANNOTATIONS_MASK_PATH = '../../resources/sentinel/Sentinel2/manual_annotated/gdal_croped/annotations'
METHODS_PREDICTIONS_PATH = '../../resources/sentinel/Sentinel2/manual_annotated/gdal_croped/methods'

METHODS = [
    'KatoNakamura', 'Liu', 'Murphy'
]

if __name__ == '__main__':

    images_paths = glob(os.path.join(MANUAL_ANNOTATIONS_IMAGE_PATH, '*.tif'))
    print('Num images: ', len(images_paths))

    for image_path in tqdm(images_paths):
        with rasterio.open(image_path) as src:
            img = src.read((6,5,4))

        # Se houver valor diferente de zero a imagem possui pixel valido e não precisa ser excluída
        if img.max() != 0:
            continue

        image_patch_name = os.path.basename(image_path)
        mask_name = image_patch_name.replace('_20m_stack_', '_20m_stack_maskf_')

        annotation_path = os.path.join(MANUAL_ANNOTATIONS_MASK_PATH, mask_name)
        
        if os.path.exists(annotation_path):
            os.remove(annotation_path)
        
        for method in METHODS:

            prediction_name = image_patch_name.replace('_20m_stack_', '_mask_')
            prediction_path = os.path.join(METHODS_PREDICTIONS_PATH, method, prediction_name)

            if os.path.exists(prediction_path):
                os.remove(prediction_path)

        # remove a imagem por ultimo, se houver erro a imagem pode ser reprocessada
        os.remove(image_path)


