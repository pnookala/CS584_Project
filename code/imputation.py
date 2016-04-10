import sys, getopt, ntpath, os
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sklearn.preprocessing as skp
from sklearn import linear_model
from myUtil import *

debug = False


# debug = False

# clear_data_random is used to "manually" delete data in the target array.
# This is done for testing purposes so that a consistent test can be performed on the resulting data.
#
# data        - The input data source
# row_rate    - [0 to 1] Percentage of rows to be affected
# col_rate    - How mnay of the columns should be cleared in the target row
# seed        - Used to control the randomization, used for debugging
def clear_data_random(data, row_rate=1., col_rate=1, seed=None):
    if (row_rate < 0) | (row_rate > 1):
        raise ValueError("row_rate of " + str(row_rate) + " is invalid. Valid range is 0 - 1 inclusive")

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
            data[indices[i], col%M] = math.nan

    return data


# CIMV: Current Imputable Missing Value
# Ri: the impact weight of the ith instance
# Wi: the mutual information between the ith attribute and the class label
# Sign(i,j): the significance of the missing value which is located in the ith instance and on the jth attribute
class Imputation:
    def __init__(self):
        self.dtype = np.float64
        self.alpha = 1
        ''

    def estimate_values(self, data):
        # First find parameters
        M, N = data.shape
        instance_rate = np.zeros((M), dtype=self.dtype)  # 1/R_i
        impact_weight = np.zeros((M), dtype=self.dtype)  # R_i
        attribute_rate = np.zeros((N), dtype=self.dtype)  # 1/W_i
        mutual_info = np.zeros((N), dtype=self.dtype)  # W_i

        average = np.zeros((N), dtype=self.dtype)
        count = np.zeros((N), dtype=self.dtype)

        missing_indices = []
        for i in range(M):
            for j in range(N):
                if math.isnan(data[i, j]):
                    instance_rate[i] += 1
                    attribute_rate[j] += 1
                    missing_indices.append((i, j))
                else:
                    average[j] += data[i, j]
                    count[j] += 1

            if instance_rate[i] > 0:
                impact_weight[i] = 1 / instance_rate[i]

        for j in range(N):
            average[j] /= count[j]
            if attribute_rate[j] > 0:
                mutual_info[j] = 1 / attribute_rate[j]

        self.impact_weight = impact_weight
        self.mutual_info = mutual_info
        n_missing = len(missing_indices)

        # then calculate the significance of each missing value (using the indices we already found)
        significance = np.ndarray((n_missing), dtype=self.dtype)

        for k in range(n_missing):
            i, j = missing_indices[k]
            significance[k] = self.sign(i, j)
            data[i, j] = average[j]  # Impute the average values as a first-pass guess

        # Then find the order to impute the data
        sorted_indices = np.argsort(significance)

        if debug:
            myshow(average, "Average")
            myshow(impact_weight.reshape(-1, 1), "impact_weight, R_i")
            myshow(mutual_info, "mutual_info, W_i")

            myshow(sorted_indices, "sorted_indices")
            myshow(significance, "significance", maxlines=20)
            myshow(missing_indices, "missing_indices")

        # CIMV is now equivalent to:
        #   missing_indices[sorted_indices]
        #
        # Or rather:
        #
        #   for i in range(n_missing):
        #       CMIV_i = missing_indices[sorted_indices[i]]
        #       value = data[CMIV_i[0], CMIV_i[1]]
        #       ...

    def sign(self, i, j):
        if (self.impact_weight[i] == 0) | (self.mutual_info[j] == 0):
            return 0

        # Unweighted
        # return 1 / 2 * 1/(1 / self.impact_weight[i] + 1 / self.mutual_info[j])

        # Weighted (by alpha)
        return ((self.alpha + 1) * self.impact_weight[i] * self.mutual_info[j]) / \
               (self.impact_weight[i] + self.alpha * self.mutual_info[j])

    def get_significance(self, data):
        sign_arr = np.zeros(data.shape, dtype=self.dtype)
        M, N = data.shape
        for i in range(M):
            for j in range(N):
                if math.isnan(data[i, j]):
                    sign_arr[i, j] = self.sign(i, j)

        return sign_arr
