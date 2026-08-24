import numpy as np

from gt_guided_dino.metrics import voc_ap


def test_voc07_ap_is_one_for_perfect_curve():
    recall = np.array([0.5, 1.0])
    precision = np.array([1.0, 1.0])
    assert voc_ap(recall, precision, use_07_metric=True) == 1.0
    assert voc_ap(recall, precision, use_07_metric=False) == 1.0

