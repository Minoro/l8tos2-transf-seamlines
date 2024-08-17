import os
import pandas as pd
import sys
import argparse
import json

import config

import tensorflow as tf

from core.models import get_normalization_layer, get_model, add_bn_at_start
from core.metrics import get_model_metrics, evaluate_dataset
from core.data import get_dataset_from_paths


def read_fold_file(args, fold):    
    return pd.read_csv(os.path.join(args.csv_folds_dir, f'fold_{fold}.csv'))

def get_train_validation_test_dataset_using_normalization_layer(df_fold, normalization_layer):
    df_train, df_validation, df_test = filter_train_validation_test_dataframe(df_fold)

    train_dataset = get_dataset_from_paths(df_train['images_paths'], df_train['masks_paths'], batch_size=config.BATCH_SIZE, normalization_layer=normalization_layer, use_data_augmentation=config.USE_DATA_AUGMENTATION, shuffle=True)
    val_dataset = get_dataset_from_paths(df_validation['images_paths'], df_validation['masks_paths'], batch_size=config.BATCH_SIZE, normalization_layer=normalization_layer, use_data_augmentation=False, shuffle=False)
    test_dataset = get_dataset_from_paths(df_test['images_paths'], df_test['masks_paths'], batch_size=config.BATCH_SIZE, normalization_layer=normalization_layer, use_data_augmentation=False, shuffle=False, repeat=False)

    return train_dataset, val_dataset, test_dataset


def filter_train_validation_test_dataframe(df_fold):    
    df_fold['images_paths'] = df_fold.apply(build_image_path, axis=1)
    df_fold['masks_paths'] = df_fold.apply(build_mask_path, axis=1)

    df_train = df_fold[(df_fold['set'] == 'train')]
    df_validation = df_fold[(df_fold['set'] == 'validation')]
    df_test = df_fold[(df_fold['set'] == 'test')]

    return df_train, df_validation, df_test

def build_image_path(row):
    if row['annotation'] == 'manual':
        return os.path.join(config.MANUAL_ANNOTATIONS_IMAGE_PATH, row['patch'])

    return os.path.join(config.SEAMLINE_IMAGES_PATH, row['patch'])

def build_mask_path(row):
    if row['annotation'] == 'manual':
        return os.path.join(config.MANUAL_ANNOTATIONS_MASK_PATH, row['patch'].replace(config.ANNOTATION_IMAGE_MARKER,  config.ANNOTATION_MASK_MARKER))
    
    return ''

def define_output_folder_name(args):
    identification = args.identification
    if identification != '':
        identification += '_'

    output_folder_name = f'{identification}{config.BASE_MODEL}_{args.normalization}_{str(args.quantification)}_Scratch'

    return output_folder_name

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Transfer Learning do Landsat-8 para o Sentinel-2.')
    
    parser.add_argument('--model', action="store", choices=['unet', 'deeplabv3+', 'SegFormerB0'], default=[config.MODEL], nargs='*', help="Modelo(s) para treinar" )  
    parser.add_argument('--csv-folds-dir', action='store', type=str, default=config.CSV_FOLDS_BASE_PATH, help="Diretório com os arquivos CSV com a divisão dos folds")
    parser.add_argument('--fold', action="store", choices=list(range(1, config.NUM_FOLDS+1)), default=list(range(1, config.NUM_FOLDS+1)), type=int, nargs='*', help="Número(s) do(s) fold(s) para ser(em) avaliado(s)")
    parser.add_argument('--normalization', action="store", choices=['no-bn'], default=config.NORMALIZATION_MODE, help="Modo de normalização da imagem de entrada da rede")
    parser.add_argument('--quantification', action="store", type=float, default=config.SENTINEL_QUANTIFICATION_VALUE, help='Valor de quantificação das imagens do Sentinel-2. Se houver normalização, a imagem será dividida por esse valor')
    parser.add_argument('--epochs', action="store", type=int, default=config.EPOCHS, help="Número de épocas para treinar")
    parser.add_argument('--gpu', action="store", type=str, default=str(config.CUDA_DEVICE), help="Dispositivo GPU usado")
    
    parser.add_argument('--identification', action="store", type=str, default=config.IDENTIFICATION_TRAIN_PREFIX, help="Prefixo da pasta para identificar o experimento sendo executado.")
    args = parser.parse_args()

    output_folder_name = define_output_folder_name(args)
    normalization_layer = get_normalization_layer(args.normalization, args.quantification)    
    for model_name in args.model:
        for fold in args.fold:    
            
            output_dir = os.path.join(config.OUTPUT_RESULTS_TRAIN_FROM_SCRATCH_PATH, output_folder_name, model_name, str(fold))
            os.makedirs(output_dir, exist_ok=True)

            output_results_file = os.path.join(output_dir, f"train_from_scratch_results.json")
            if os.path.exists(output_results_file):
                print(f'Ignoring model: {model_name} - Fold: {fold} - Pretrained: {config.BASE_MODEL}')
                continue
            

            # Limpa a sessão do tensor flow psara carregar um novo modelo
            tf.keras.backend.clear_session()
            
            # Carrega o modelo e congela as camadas
            model = get_model(model_name, config.IMAGE_SHAPE)

            model.compile(
                optimizer=tf.keras.optimizers.Adam(config.LR),
                loss=config.LOSS_FUNCTION,
                metrics=[get_model_metrics()],
            )
        
            df_fold = read_fold_file(args, fold)

            df_train, df_validation, df_test = filter_train_validation_test_dataframe(df_fold)
            train_dataset, val_dataset, test_dataset = get_train_validation_test_dataset_using_normalization_layer(df_fold, normalization_layer)

            if args.epochs == 0:
                print('Treino desabilitado, avaliando o modelo...')
                print(f'Evaluating model: {model_name} - Fold: {fold} - Pretrained: {config.BASE_MODEL}')
                results = evaluate_dataset(model, test_dataset)
                print(results)
                continue
                
            print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
            print(f'Modelo: {model_name} - Fold: {fold} - Pesos: {config.BASE_MODEL}')
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config.EARLY_STOP_PATIENCE, restore_best_weights=config.EARLY_STOP_RESTORE_BEST, verbose=1)
            history = model.fit(
                train_dataset, 
                validation_data=val_dataset, 
                steps_per_epoch=len(df_train) // config.BATCH_SIZE,
                validation_steps=len(df_validation) // config.BATCH_SIZE,
                epochs=args.epochs, 
                callbacks=[es]
            )

            # Save the history file with the loss value for each epoch
            with open(os.path.join(output_dir, f"train_from_scratch_history.json"), "w") as f:
                json.dump(history.history, f, default=str)

            del history

            output_weights_dir = os.path.join(config.OUTPUT_WEIGHTS_TRAIN_FROM_SCRATCH_PATH, output_folder_name, model_name, str(fold))
            os.makedirs(output_weights_dir, exist_ok=True)
            
            model.save(os.path.join(output_weights_dir, f'model.keras'))

            # Release the memory
            del train_dataset
            del val_dataset

            print(f'Evaluating model: {model_name} - Fold: {fold} - Pretrained: {config.BASE_MODEL}')
            results = evaluate_dataset(model, test_dataset)
            print(results)
            with open(output_results_file, "w") as f:
                json.dump(results, f)
            del results

            del model
            print(f'Resultados salvos no arquivo: {output_results_file}')
        
    print('Done!')