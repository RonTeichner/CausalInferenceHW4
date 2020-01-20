import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import json
import os
import pickle
from CausalInferenceHW4_func import *

enable_IPW = True
enableMatching = True
enable_S_learner = True
enable_T_learner = True

datasetNums = [1, 2]

for datasetNum in datasetNums:
    X, T, Y = read_csv_data(datasetNum)
    scaler, X_train_scaled, X_test_scaled, X_train, T_train, Y_train, T_test, Y_test = create_sets(X, T, Y)

    ########################################## Propensity score #######################################################
    if enable_IPW:
        mu_ATT_IPW = calc_ATT_IPW(X_train_scaled, T_train, Y_train, X_test_scaled, T_test, Y_test, datasetNum, scaler, X, T, Y)
        print(f'dataset{datasetNum}: IPW: {mu_ATT_IPW}')

    ########################################## matching #######################################################
    if enableMatching:
        mu_ATT_matching = calc_ATT_matching(datasetNum, scaler, X, T, Y)
        print(f'dataset{datasetNum}: matching: {mu_ATT_matching}')

    ########################################## S-learner #######################################################
    if enable_S_learner:
        mu_ATT_sLearner = calc_ATT_sLearner(X_train_scaled, X_test_scaled, T_test, Y_test, T_train, Y_train, scaler, datasetNum, X, T)
        print(f'dataset{datasetNum}: s-Learner: {mu_ATT_sLearner}')

    ########################################## T-learner #######################################################
    if enable_T_learner:
        mu_ATT_tLearner = calc_ATT_tLearner(scaler, X_train_scaled, T_train, T_test, X_test_scaled, Y_test, Y_train, datasetNum, X, T)
        print(f'dataset{datasetNum}: t-Learner: {mu_ATT_tLearner}')