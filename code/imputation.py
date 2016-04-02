import sys, getopt, ntpath, os
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sklearn.preprocessing as skp
from sklearn import linear_model


# clear_data_random is used to "manually" delete data in the target array.
# This is done for testing purposes so that a consistent test can be performed on the resulting data.
#
# data        - The input data source
# row_rate    - [0 to 1] Percentage of rows to be affected
# col_rate    - How mnay of the columns should be cleared in the target row
# seed        - Used to control the randomization, used for debugging
def clear_data_random(data, row_rate=1., col_rate=1, seed=None):
    if (row_rate < 0) | (row_rate > 1):
        raise "row_rate of " + str(row_rate) + " is invalid. Valid range is 0 - 1 inclusive"

    if len(data.shape) != 2:
        raise "Can only clear data in a 2-D array"

    M, N = data.shape
    np.random.seed(seed=seed)
    # First create randomized indices by shuffling
    indices = np.arange(0, M)
    end = M
    for i in range(M):
        k = np.random.randint(0, end)
        end -= 1
        t = indices[k]
        indices[k] = indices[end]
        indices[end] = t

    # Then for <row_rate>% of the rows
    for i in range(math.ceil(M * row_rate)):
        # Clear out up to <col_rate> values from each row
        for j in range(col_rate):
            col = np.random.randint(0, N)
            data[indices[i], col] = math.nan

    return data


class Imputation:
    def __init__(self):
        ''

    def estimate_values(self, data):
        ''
