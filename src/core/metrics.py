import numpy as np
import tensorflow as tf
from tqdm import tqdm


def statistics(y_true, y_pred):
    y_pred_neg = 1 - y_pred
    y_expected_neg = 1 - y_true

    tp = np.sum(y_pred * y_true)
    tn = np.sum(y_pred_neg * y_expected_neg)
    fp = np.sum(y_pred * y_expected_neg)
    fn = np.sum(y_pred_neg * y_true)
    return tn, fp, fn, tp

def evaluate(annotations, predictions, verbose=True):

    if len(annotations) != len(predictions):
        raise 'Número de predições e anotações diferente'

    y_pred_all_v1 = []
    y_true_all_v1 = []

    count_fire_pixel_mask = 0
    count_fire_pixel_pred = 0

    i = 0
    for y_true, y_pred in tqdm(zip(annotations, predictions), total=len(predictions)):
        i += 1

        y_true = np.array(y_true != 0, dtype=np.uint8)
        y_pred = np.array(y_pred > 0.5, dtype=np.uint8)

        y_pred = y_pred.flatten() 
        y_true = y_true.flatten()

        y_pred_all_v1.append(y_pred)
        y_true_all_v1.append(y_true)


        count_fire_pixel_mask += np.sum(y_true)
        count_fire_pixel_pred += np.sum(y_pred)

    y_pred_all_v1 = np.array(y_pred_all_v1, dtype=np.uint8)
    y_pred_all_v1 = y_pred_all_v1.flatten()

    y_true_all_v1 = np.array(y_true_all_v1, dtype=np.uint8)
    y_true_all_v1 = y_true_all_v1.flatten()

    if verbose:
        print('True Fire:', count_fire_pixel_mask)
        print('Fire Pred:', count_fire_pixel_pred)
        print('y_true_all_v1 shape: ', y_true_all_v1.shape)
        print('y_pred_all_v1 shape: ', y_pred_all_v1.shape)

    tn, fp, fn, tp = statistics(y_true_all_v1, y_pred_all_v1)

    P = float(tp)/(tp + fp)
    R = float(tp)/(tp + fn)
    IoU = float(tp)/(tp+fp+fn)
    Acc = float((tp+tn))/(tp+tn+fp+fn)
    if (P + R) == 0:
        F = 0.0
    else:
        F = (2 * P * R)/(P + R)

    if verbose:    
        print('Statistics3 (tn, fp, fn, tp):', tn, fp, fn, tp)
        print('P: :', P, ' R: ', R, ' IoU: ', IoU, ' Acc: ', Acc, ' F-score: ', F, 'TP: ', tp, 'TN:', tn, 'FP:', fp, 'FN:', fn)

    results = {
        'P' : P,
        'R' : R,
        'IoU': IoU,
        'F-score': F,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }

    return results


def evaluate_dataset(model, dataset, verbose=True):

    predictions = model.predict(dataset)
    
    annotations = []
    for img, annotation in dataset:
        for mask in annotation:
            annotations.append(mask)
        
    annotations = np.array(annotations)
    if verbose:
        print('Número de anotaçẽos: ', len(annotations))
        print('Número de predições', len(predictions))

    return evaluate(annotations, predictions, verbose)



def get_model_metrics():
    metrics = {
        'P': tf.keras.metrics.Precision(),
        'R': tf.keras.metrics.Recall(),
    }

    return metrics