"""
Cria os folds para teste treino e validação dos modelos.
O código verifica as imagens anotadas para verificar quais tem fogo e quais não tem, também verifica as imagens de costura
isso é utilizado para gerar os folds de forma estratificada. Assim, é mantida a proporção de cada tipo de imagem nos folds.
Para desativar a estratificação basta alterar o arquivo de configuração.
É possível reservar amostras para validação, se ativo, será reservado um conjunto do mesmo tamanho do conjunto de teste.
"""
import sys
import os
import pandas as pd
from glob import glob
import rasterio
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

# Configurações do script (arquivo config.py)
import config

def gerar_csv_com_idenficacao_de_fogo_e_seamline_por_patch():
    annotations_paths = sorted(glob(os.path.join(config.MANUAL_ANNOTATIONS_MASK_PATH, '*.tif')))
    
    print('Num. Sentinel-2 images: ', len(annotations_paths))
    print('Counting fire pixels in images...')
    data = []
    for annotation_path in tqdm(annotations_paths):
        with rasterio.open(annotation_path) as src:
            annotation = src.read(1)

        patch_name = os.path.basename(annotation_path).replace('_maskf_', '_')

        num_fire_pixels = annotation.sum()
        data.append({
            'patch': patch_name,
            'annotation': 'manual',
            'num_fire_pixels': num_fire_pixels,
            'category': 'fire' if num_fire_pixels > 0 else 'no-fire',
        })

    print('Checking seam-line images...')    
    seamlines_paths = sorted(glob(os.path.join(config.SEAMLINE_IMAGES_PATH, '*.tif')))
    print('Num. seam-line patches: ', len(seamlines_paths))
    for seamline_path in tqdm(seamlines_paths):
        patch_name = os.path.basename(seamline_path)
        data.append({
            'patch': patch_name,
            'annotation': 'seamline',
            'num_fire_pixels': 0,
            'category': 'seamline',
        })

    os.makedirs(os.path.dirname(config.CSV_NUM_FIRE_PIXELS_PER_PATCH_PATH), exist_ok=True)
    df_patches = pd.DataFrame(data)
    df_patches.to_csv(config.CSV_NUM_FIRE_PIXELS_PER_PATCH_PATH)

    return df_patches

def show_num_patches_in_each_category(df_patches):

    print('Num. patches with fire: ', len(df_patches[ df_patches['category'] == 'fire' ]))
    print('Num. patches Annotations without fire: ', len(df_patches[ (df_patches['annotation'] == 'manual') & (df_patches['category'] == 'no-fire') ]))
    print('Num. patches seam-lines: ', len(df_patches[ df_patches['annotation'] == 'seamline' ]))


if __name__ == '__main__':

    if config.OVERRIDE_CSV_NUM_FIRE_PIXELS_PER_PATCH or not os.path.exists(config.CSV_NUM_FIRE_PIXELS_PER_PATCH_PATH):
        gerar_csv_com_idenficacao_de_fogo_e_seamline_por_patch()

    df_patches = pd.read_csv(config.CSV_NUM_FIRE_PIXELS_PER_PATCH_PATH, index_col=0)
    show_num_patches_in_each_category(df_patches)

    X_patches = df_patches.drop('category', axis=1)
    y_categories = df_patches['category']
    
    kfold = KFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    if config.STRATIFIED_FOLDS:
        kfold = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)

    os.makedirs(config.CSV_FOLDS_BASE_PATH, exist_ok=True)
    folds = []
    for _, test_index in kfold.split(X_patches, y_categories):
        folds.append(test_index)
    
    for k, fold in enumerate(folds):
        print('Fold', k+1)

        df_val = pd.DataFrame(columns=X_patches.columns)
        if config.GENERATE_VALIDATION_FOLD:
            k_val = (k + 1) % len(folds)

            df_val = df_patches.iloc[folds[k_val]].copy()

        df_test = df_patches.iloc[fold].copy()
        df_train = df_patches[ (~df_patches.index.isin(df_test.index)) & (~df_patches.index.isin(df_val.index) )].copy()

        print('Fold: ', k+1)
        print('Train:')
        show_num_patches_in_each_category(df_train)
        print('\n\nValidation:')
        show_num_patches_in_each_category(df_val)
        print('\n\nTest')
        show_num_patches_in_each_category(df_test)
        print('-'*80)
        
        df_train['set'] = 'train'
        df_val['set'] = 'validation'
        df_test['set'] = 'test'

        df_fold = pd.concat((df_train, df_val, df_test))
        df_fold['fold'] = k+1
        
        df_fold.to_csv(os.path.join(config.CSV_FOLDS_BASE_PATH, f'fold_{k+1}.csv'))


    print('Done!')

