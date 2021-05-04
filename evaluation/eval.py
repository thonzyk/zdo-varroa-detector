import argparse

from matplotlib.pyplot import imread
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')

parser = argparse.ArgumentParser(description='Generates sequences for labeling task')
parser.add_argument('label_file', metavar='LABEL', type=str, help='Path to .npy file containing label mask.')
parser.add_argument('pred_file', metavar='PRED', type=str, help='Path to .npy file containing prediction mask.')

if __name__ == '__main__':
    args = parser.parse_args()
    y_true = imread(args.label_file)
    y_true = y_true[:, :, 1] > 0.1

    y_pred = np.load(args.pred_file)

    y_true = y_true.astype('float32')
    y_pred = y_pred.astype('float32')

    y_show = np.concatenate([np.expand_dims(y_true, axis=2), np.expand_dims(y_pred, axis=2),
                             np.zeros((y_true.shape[0], y_true.shape[1], 1))], axis=2)

    y_true = y_true.reshape((y_true.size,))
    y_pred = y_pred.reshape((y_pred.size,))

    f1_1 = f1_score(y_true, y_pred)
    f1_2 = f1_score(1.0 - y_true, 1.0 - y_pred)
    score = (f1_1 + f1_2) / 2

    cf_mat = confusion_matrix(y_true, y_pred)
    cf_mat_2 = confusion_matrix(1.0 - y_true, 1.0 - y_pred)
    print(score)
    print(cf_mat)

    plt.figure()
    plt.imshow(y_show)
    plt.show()

    # plt.show()
