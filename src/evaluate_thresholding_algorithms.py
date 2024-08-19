import os
import sys
import pandas as pd
import rasterio
from glob import glob
from tqdm import tqdm
import numpy as np
import argparse

from core.metrics import evaluate
import config

cache_annotations = {}
cache_predictions = {}

def carregar_anotacoes_manuais():
    print('Carregando anotações manuais')
    annotations_paths = glob(os.path.join(config.MANUAL_ANNOTATIONS_MASK_PATH, '*.tif'))

    for annotation_path in tqdm(annotations_paths, total=len(annotations_paths)):
        annotation_name = os.path.basename(annotation_path)
        with rasterio.open(annotation_path) as src:
            annotation = src.read(1)

        cache_annotations[annotation_name] = annotation

    print('Anotaçõse carregadas')

def carregar_predicoes_do_metodo(method):
    cache_predictions = {}
    print(f'Carregando predições do método {method}')

    predictions_paths = glob(os.path.join(config.METHODS_PREDICTIONS_ANNOTATED_PATH, method, '*.tif')) 
    predictions_paths += glob(os.path.join(config.METHODS_PREDICTIONS_SEAMLINE_PATH, method, '*.tif'))

    for prediction_path in tqdm(predictions_paths, total=len(predictions_paths)):
        patch_name = os.path.basename(prediction_path)

        with rasterio.open(prediction_path) as src:
            pred = src.read(1)

        cache_predictions[patch_name] = pred

    print('Predições carregadas!')


def calcuar_metricas_do_metodo_para_fold(method, df_fold):
    print(f'Avaliando método: {method}')

    df_test = df_fold[ df_fold['set'] == 'test' ]

    annotations = []
    predictions = []

    print('Carregando imagens do fold...')
    for i, row in tqdm(df_test.iterrows(), total=len(df_test)):
        annotation = load_annotation(row['patch'], row['annotation'])
        prediction = load_prediction(method, row['mask'], row['annotation'])

        annotations.append(annotation)
        predictions.append(prediction)

    print('Calculando métricas...')
    return evaluate(annotations, predictions)

def load_annotation(patch, type):
    if type == 'seamline':
        return np.zeros((config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]))

    patch = patch.replace(config.ANNOTATION_IMAGE_MARKER, config.ANNOTATION_MASK_MARKER)
    annotation_path = os.path.join(config.MANUAL_ANNOTATIONS_MASK_PATH, patch)
    with rasterio.open(annotation_path) as src:
        return src.read(1)


def load_prediction(method, patch, type):
    if patch in cache_predictions:
        return cache_predictions[patch]

    prediction_path = os.path.join(config.METHODS_PREDICTIONS_ANNOTATED_PATH, method, patch)
    if type == 'seamline':
        prediction_path = os.path.join(config.METHODS_PREDICTIONS_SEAMLINE_PATH, method, patch)

    with rasterio.open(prediction_path) as src:
        pred = src.read(1)

    return pred

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Avalia o algoritmo de limiarização.')
    
    parser.add_argument('--method', action="store", choices=config.THRESHOLDING_METHODS, default=config.THRESHOLDING_METHODS, nargs='*', help="Algoritmo para ser avaliado.")
    parser.add_argument('--fold', action="store", choices=list(range(1, config.NUM_FOLDS+1)), default=list(range(1, config.NUM_FOLDS+1)), type=int, nargs='*', help="Número(s) do(s) fold(s) para ser(em) avaliado(s)")

    args = parser.parse_args()

    os.makedirs(config.OUTPUT_RESULTS_METHODS_PREDICTIONS_PATH, exist_ok=True)
    if config.PRELOAD_ANNOTATIONS_TO_COMPARE_WITH_METHODS_PREDICTIONS:
        carregar_anotacoes_manuais()

    results_data = []
    for method in args.method:
        cache_predictions = {}
        if config.PRELOAD_METHODS_PREDICTIONS:
            carregar_predicoes_do_metodo(method)

        for fold in args.fold:
            print(f'Avaliando: {method} - Fold: {fold}')
            path_fold_file = os.path.join(config.CSV_FOLDS_BASE_PATH, f'fold_{fold}.csv')
            
            df_fold = pd.read_csv(path_fold_file)
            df_fold['mask'] = df_fold['patch'].str.replace('_20m_stack_', '_mask_').str.replace('_b12b11b8a_', '_mask_')
            
            results = calcuar_metricas_do_metodo_para_fold(method, df_fold)

            results['fold'] = fold
            results['method'] = method

            df_results = pd.DataFrame([results])
            df_results.to_csv(os.path.join(config.OUTPUT_RESULTS_METHODS_PREDICTIONS_PATH, f'results_{method}_fold_{fold}.csv'), index=False)

            results_data.append(results)

    df_results = pd.DataFrame(results_data)
    df_results.to_csv(os.path.join(config.OUTPUT_RESULTS_METHODS_PREDICTIONS_PATH, 'results.csv'), index=False)

    print( df_results.groupby('method').agg({'mean', 'std'}) )

    df_results.groupby('method').agg({'mean', 'std'}).reset_index().to_csv(os.path.join(config.OUTPUT_RESULTS_METHODS_PREDICTIONS_PATH, 'results_summary.csv'), index=False)
    print('Done!')  