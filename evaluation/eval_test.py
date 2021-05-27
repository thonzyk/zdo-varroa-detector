import unittest

import numpy as np
from matplotlib.pyplot import imread
from sklearn.metrics import f1_score


def get_score(label_file, pred_file):
    """Get F1 score for pair of labeled and prediction images."""
    y_true = imread(label_file)
    y_true = y_true[:, :, 1] > 0.1
    y_pred = np.load(pred_file)
    y_true = y_true.astype('float32')
    y_pred = y_pred.astype('float32')
    y_true = y_true.reshape((y_true.size,))
    y_pred = y_pred.reshape((y_pred.size,))

    f1_1 = f1_score(y_true, y_pred)
    f1_2 = f1_score(1.0 - y_true, 1.0 - y_pred)
    score = (f1_1 + f1_2) / 2

    return score


class TestScore(unittest.TestCase):
    def test_f1_on_mix_image(self):
        label_file = '../images/concat_mask.png'
        pred_file = 'out.npy'
        score = get_score(label_file, pred_file)
        assert score > 0.63
        assert score <= 1.0
