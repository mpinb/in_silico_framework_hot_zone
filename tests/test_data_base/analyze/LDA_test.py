from data_base.analyze.LDA import make_groups_equal_size
import numpy as np


def test_make_groups_equal_size():
    assert len(
        make_groups_equal_size(
            np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]),
            np.array([0, 0, 1, 1]))[0]) == 4
    
    assert len(
        make_groups_equal_size(
            np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]),
            np.array([0, 0, 0, 1]))[0]) == 2