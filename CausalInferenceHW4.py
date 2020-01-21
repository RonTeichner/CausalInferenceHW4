import pandas as pd
import csv
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
nIters = 30

dataRes = list()
ps_scores_all = list()

for i in range(nIters):
    print(f'starting iter no. {i}')
    dataRes.append(list())
    ps_scores_all.append(list())
    for dataSetIdx, datasetNum in enumerate(datasetNums):
        dataRes[i].append(list())
        ps_scores_all[i].append(list())

        if datasetNum == 1:
            lowPsTh = 0.33  # from the histogram it seems that for overlap we should only keep the patients whose propensity score is above 0.3:
            matchingMahalanobisThr = 7.5  # for good matching let's match only on the treated that has a control match with a mahalanibis dist smaller than 7.5. We remain with about 80% of treated patients:
        elif datasetNum == 2:
            lowPsTh = 0.16  # from the histogram it seems that for overlap we should only keep the patients whose propensity score is above 0.16
            matchingMahalanobisThr = 7.5  # for good matching let's match only on the treated that has a control match with a mahalanibis dist smaller than 7.5. We remain with about 80% of treated patients:

        X, T, Y = read_csv_data(datasetNum)
        scaler, X_train_scaled, X_test_scaled, X_train, T_train, Y_train, T_test, Y_test = create_sets(X, T, Y)

        ########################################## IPW ############################################################
        if enable_IPW:
            mu_ATT_IPW, ps_scores = calc_ATT_IPW(X_train_scaled, T_train, Y_train, X_test_scaled, T_test, Y_test, datasetNum, scaler, X, T, Y, lowPsTh)
            dataRes[i][dataSetIdx].append(mu_ATT_IPW)
            ps_scores_all[i][dataSetIdx].append('data%d' % datasetNum)
            for psIdx in range(ps_scores.shape[0]):
                ps_scores_all[i][dataSetIdx].append(ps_scores[psIdx][0])
            print(f'dataset{datasetNum}: IPW: {mu_ATT_IPW}')

        ########################################## S-learner #######################################################
        if enable_S_learner:
            mu_ATT_sLearner = calc_ATT_sLearner(X_train_scaled, X_test_scaled, T_test, Y_test, T_train, Y_train, scaler, datasetNum, X, T)
            dataRes[i][dataSetIdx].append(mu_ATT_sLearner)
            print(f'dataset{datasetNum}: s-Learner: {mu_ATT_sLearner}')

        ########################################## T-learner #######################################################
        if enable_T_learner:
            mu_ATT_tLearner = calc_ATT_tLearner(scaler, X_train_scaled, T_train, T_test, X_test_scaled, Y_test, Y_train, datasetNum, X, T)
            dataRes[i][dataSetIdx].append(mu_ATT_tLearner)
            print(f'dataset{datasetNum}: t-Learner: {mu_ATT_tLearner}')

        ########################################## matching ########################################################
        if enableMatching:
            if i == 0:
                mu_ATT_matching = calc_ATT_matching(datasetNum, scaler, X, T, Y, matchingMahalanobisThr)
            else: # matching is deterministic
                mu_ATT_matching = dataRes[0][dataSetIdx][3]

            dataRes[i][dataSetIdx].append(mu_ATT_matching)
            print(f'dataset{datasetNum}: matching: {mu_ATT_matching}')

        ########################################## final estimate ###################################################
        if datasetNum == 1:
            mu_ATT_final = 0
        elif datasetNum == 2:
            mu_ATT_final = 0

        dataRes[i][dataSetIdx].append(mu_ATT_final)
        print(f'dataset{datasetNum}: final estimate: {mu_ATT_final}')

# fuse data from iterations:
nMethods = 4
results = np.zeros((nIters, len(datasetNums), nMethods))
medianResults = np.zeros((len(datasetNums), nMethods+1))
stdResults = np.zeros((len(datasetNums), nMethods))
indexClosestToMedian = np.zeros((len(datasetNums), nMethods))

for methodIdx in range(nMethods):
    for dataSetIdx, datasetNum in enumerate(datasetNums):
        for i in range(nIters):
            results[i, dataSetIdx, methodIdx] = dataRes[i][dataSetIdx][methodIdx]
        medianResults[dataSetIdx, methodIdx] = np.median(results[:, dataSetIdx, methodIdx])
        stdResults[dataSetIdx, methodIdx] = np.std(results[:, dataSetIdx, methodIdx])
        indexClosestToMedian[dataSetIdx, methodIdx] = np.argmin(np.abs(medianResults[dataSetIdx, methodIdx] - results[:, dataSetIdx, methodIdx]))

bestIPWIdx = np.zeros(len(datasetNums))
for dataSetIdx, datasetNum in enumerate(datasetNums):
    bestIPWIdx[dataSetIdx] = indexClosestToMedian[dataSetIdx, 0]

resultsDict = {'results': results, 'medianResults': medianResults, 'stdResults': stdResults, 'indexClosestToMedian': indexClosestToMedian, 'bestIPWIdx': bestIPWIdx}
pickle.dump(resultsDict, open('./resultsDict', "wb"))

ATT_res = {'Type': [1, 2, 3, 4, 5],
        'data1': medianResults[0],
        'data2': medianResults[1]}
df = pd.DataFrame(ATT_res, columns=['Type', 'data1', 'data2'])
df.to_csv (r'./ATT_results.csv', index=None, header=True) #Don't forget to add '.csv' at the end of the path

with open("./models_propensity.csv", "w") as fp:
    wr = csv.writer(fp, dialect='excel')
    wr.writerow(ps_scores_all[int(bestIPWIdx[0])][0])
    wr.writerow(ps_scores_all[int(bestIPWIdx[1])][1])
    fp.close()