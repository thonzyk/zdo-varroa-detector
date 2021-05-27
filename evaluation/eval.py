import argparse

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import imread
from sklearn.metrics import f1_score, confusion_matrix
import os

matplotlib.use('Qt5Agg')

parser = argparse.ArgumentParser(description='Generates sequences for labeling task')
parser.add_argument('label_dir', metavar='LABEL', type=str,
                    help='Path to directory containing label masks in png format.')
parser.add_argument('pred_dir', metavar='PRED', type=str,
                    help='Path to directory containing prediction masks in npy format.')

if __name__ == '__main__':
    args = parser.parse_args()

    label_names = next(os.walk(args.label_dir))
    pred_names = next(os.walk(args.pred_dir))
    names = [el.split('.')[0] for el in label_names]

    score = 0.0

    for name in names:
        label_file = name + '.png'
        pred_file = name + '.npy'

        y_true = imread(label_file)
        y_true = y_true[:, :, 1] > 0.1

        y_pred = np.load(pred_file)

        y_true = y_true.astype('float32')
        y_pred = y_pred.astype('float32')

        y_show = np.concatenate([np.expand_dims(y_true, axis=2), np.expand_dims(y_pred, axis=2),
                                 np.zeros((y_true.shape[0], y_true.shape[1], 1))], axis=2)

        y_true = y_true.reshape((y_true.size,))
        y_pred = y_pred.reshape((y_pred.size,))

        f1_1 = f1_score(y_true, y_pred)
        f1_2 = f1_score(1.0 - y_true, 1.0 - y_pred)
        sample_score = (f1_1 + f1_2) / 2
        print(f"Sample F1-score: {sample_score}")
        score += sample_score

    score /= len(names)
    print(f"Mean total F1-score: {score}")
