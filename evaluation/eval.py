import argparse

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
    y_true = np.load(args.label_file).astype('bool')
    y_pred = np.load(args.pred_file)

    # plt.figure()
    # plt.imshow(y_true)
    # plt.figure()
    # plt.imshow(y_pred)

    y_true = y_true.reshape((y_true.size,))
    y_pred = y_pred.reshape((y_pred.size,))

    score = f1_score(y_true, y_pred)

    cf_mat = confusion_matrix(y_true, y_pred)
    print(score)
    print(cf_mat)

    # plt.show()
