import sys
import os
import pandas as pd
import json
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import argparse
from datetime import datetime

sys.path.append('../')
import landsat_config as config
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.CUDA_DEVICE)
from core.data import get_landsat_images_dataset_and_num_images_from_config_and_args
from core.models import get_model

TRAIN_SET_NAME = 'train'
VALIDATION_SET_NAME = 'val'

def train_model(args):

    bands_names = ''.join(['B'+str(b) for b in args.bands])
    output_dir = os.path.join(config.LANDSAT_OUTPUT_DIR, args.model, args.mask, bands_names)
    os.makedirs(output_dir, exist_ok=True)

    train_ds, num_train_images = get_landsat_images_dataset_and_num_images_from_config_and_args(config, args, TRAIN_SET_NAME)
    val_ds, num_val_images = get_landsat_images_dataset_and_num_images_from_config_and_args(config, args, VALIDATION_SET_NAME)

    input_shape = (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], len(args.bands))
    model = get_model(args.model, input_shape=input_shape, num_classes=1)
    metrics = {
        'P': tf.keras.metrics.Precision(),
        'R': tf.keras.metrics.Recall(),
    }

    model.compile(optimizer=tf.keras.optimizers.Adam(config.LR), loss = 'binary_crossentropy', metrics=metrics.values())
    model.summary()
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=args.early_stoping_patience, restore_best_weights=True)
    checkpoint_name = os.path.join(output_dir, config.CHECKPOINT_MODEL_NAME.format(args.model, args.mask))
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', save_freq=config.CHECKPOINT_PERIOD)

    print('Training using {}...'.format(args.mask))
    started_at = datetime.now()
    history = model.fit(
        train_ds,
        steps_per_epoch=num_train_images // args.batch_size,
        validation_data=val_ds,
        validation_steps=num_val_images // args.batch_size,
        callbacks=[checkpoint, es],
        # callbacks=[es],
        epochs=args.epochs
    )
    finished_at = datetime.now()
    print('Train finished!')

    print('Saving weights')
    model_weights_output = os.path.join(output_dir, config.FINAL_WEIGHTS_OUTPUT.format(args.model, args.mask, ''.join([str(b) for b in args.bands])))
    model.save_weights(model_weights_output)
    print("Weights Saved: {}".format(model_weights_output))

    print('Saving history...')
    out_file = open(os.path.join(output_dir, f"history_{args.model}_{args.mask}.json"), "w")
    json.dump(history.history, out_file, default=str)
    out_file.close()
    print("History Saved!")

    print('Done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a network with Landsat-8 images, using the masks informed in the arguments.')
    parser.add_argument('--model', action="store", choices=('unet', 'deeplabv3', 'SegFormerB0'), default=config.MODEL, help="Model to train")
    parser.add_argument('--mask', action="store", choices=('Voting', ), default=config.MASK, help="Mask to train the base model")
    parser.add_argument('--batch-size', action="store", default=config.BATCH_SIZE, type=int, help="Batch size for training")
    parser.add_argument('--lr', action="store", default=config.LR, type=float, help="Learning Rate for training")
    parser.add_argument('--epochs', action="store", default=config.EPOCHS, type=int, help="Number of epochs for training")
    parser.add_argument('--early-stopping-patience', action="store", default=config.EARLY_STOP_PATIENCE, type=int, help="Early Stopping: number of epochs to waiting until stop training if the model do not improve")
    parser.add_argument('--bands', action='store', default=config.BANDS, nargs='*', help="Bands to train the model")
    parser.add_argument('--gpu', action="store", default=str(config.CUDA_DEVICE), help="GPU to run")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_model(args)

    print('Done!')