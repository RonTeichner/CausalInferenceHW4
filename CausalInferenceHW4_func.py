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

def read_csv_data(datasetNum):
    # Read the data and create train and test sets:
    data = pd.read_csv('data%d.csv' % datasetNum, header=None)
    charColumns = np.array([2, 21, 24])
    floatColumns = np.setdiff1d(np.arange(1, 61), charColumns)

    dataMat = np.zeros((data.values.shape[0] - 1, data.values.shape[1] - 1), dtype='float')
    for colIdx in floatColumns:
        dataMat[:, colIdx - 1] = np.array(data.values[1:, colIdx], dtype='float')

    for colIdx in charColumns:
        charColumn = data.values[1:, colIdx]
        uniqueValues = np.unique(charColumn)
        for uniqueValIdx, uniqueVal in enumerate(uniqueValues):
            dataMat[np.where(charColumn == uniqueVal), colIdx - 1] = uniqueValIdx

    X, T, Y = dataMat[:, :-2], dataMat[:, -2], dataMat[:, -1]
    return X, T, Y

def create_sets(X, T, Y):
    nSamples = X.shape[0]
    nTrainSamples = int(np.round(0.8 * nSamples))
    trainIndexes = np.sort(np.random.permutation(nSamples)[:nTrainSamples])
    testIndexes = np.setdiff1d(np.arange(1, nSamples), trainIndexes)

    X_train, T_train, Y_train = X[trainIndexes, :], T[trainIndexes], Y[trainIndexes]
    X_test, T_test, Y_test = X[testIndexes, :], T[testIndexes], Y[testIndexes]

    # scale each covariate:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    print(
        f'If I use the standardScaler correcly then the std of a feature should be 1. The std of a feature: {X_train_scaled[:, 50].std()}')

    # remove the tails outside 5std's
    goodIndexes = np.arange(X_train.shape[0])
    maxStd = 5
    for featureIdx in range(X_train_scaled.shape[1]):
        featureValues = X_train_scaled[:, featureIdx]
        goodFeatureIndexes = np.intersect1d(np.where(-maxStd < featureValues), np.where(featureValues < maxStd))
        goodIndexes = np.intersect1d(goodIndexes, goodFeatureIndexes)

    print(f'out of {X_train.shape[0]} samples we remained with {goodIndexes.shape[0]} after removing outlayers')
    X_train, T_train, Y_train = X_train[goodIndexes, :], T_train[goodIndexes], Y_train[goodIndexes]

    # scale each covariate:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return scaler, X_train_scaled, X_test_scaled, X_train, T_train, Y_train, T_test, Y_test
########################################## Propensity score #######################################################
class propensityModel(nn.Module):
    def __init__(self, covariateDim):
        super(propensityModel, self).__init__()

        self.covariateDim = covariateDim
        self.internalDim = 1
        # encoder:
        self.fc21 = nn.Linear(self.covariateDim, self.internalDim)
        self.fc22 = nn.Linear(self.internalDim, 1)
        self.fc23 = nn.Linear(self.internalDim, 1)
        #self.fc24 = nn.Linear(self.covariateDim, self.covariateDim)
        #self.fc25 = nn.Linear(self.covariateDim, 1)

        # general:
        self.logSoftMax = nn.LogSoftmax(dim=1)
        self.LeakyReLU = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x0):
        x1 = self.LeakyReLU(self.fc21(x0))
        x2 = self.fc22(x1)
        #x3 = self.fc23(x2)
        #x4 = self.LeakyReLU(self.fc24(x3))
        #x5 = self.fc25(x4)
        return self.fc21(x0)#x2

    def forward(self, x):
        return self.sigmoid(self.encode(x))

def loss_function_PS(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x)
    return BCE

def calc_ATT_IPW(X_train_scaled, T_train, Y_train, X_test_scaled, T_test, Y_test, datasetNum, scaler, X, T, Y, lowPsTh):
    X_train_scaled, T_train, Y_train = torch.tensor(X_train_scaled, dtype=torch.float), torch.tensor(T_train, dtype=torch.float), torch.tensor(Y_train, dtype=torch.float)
    X_test_scaled, T_test, Y_test = torch.tensor(X_test_scaled, dtype=torch.float), torch.tensor(T_test, dtype=torch.float), torch.tensor(Y_test, dtype=torch.float)
    nTrainSamples = X_train_scaled.shape[0]

    model_PS = propensityModel(covariateDim=X_train_scaled.shape[1]).cuda()
    trainable_params = filter(lambda p: p.requires_grad, model_PS.parameters())
    config = json.load(open('my-config.json'))
    opt_name = config['optimizer']['type']
    opt_args = config['optimizer']['args']
    #opt_args['lr'] = 1e-4
    optimizer = getattr(torch.optim, opt_name)(trainable_params, **opt_args)
    #optimizer = optim.Adam(model_PS.parameters(), lr=1e-3)

    lr_name = config['lr_scheduler']['type']
    lr_args = config['lr_scheduler']['args']
    lr_args['step_size'] = 200
    if lr_name == 'None':
        lr_scheduler = None
    else:
        lr_scheduler = getattr(torch.optim.lr_scheduler, lr_name)(optimizer, **lr_args)

    nEpochs = 200+1
    trainLoss, trainProbLoss, testLoss, testProbLoss = np.zeros(nEpochs), np.zeros(nEpochs), np.zeros(nEpochs), np.zeros(nEpochs)
    X_test_scaled, T_test = X_test_scaled.cuda(), T_test.cuda()
    X_train_scaled, T_train = X_train_scaled.cuda(), T_train.cuda()
    minTestLoss = np.inf
    for epochIdx in range(nEpochs):
        model_PS.train()
        total_loss = 0
        batchSize = 50
        nBatches = int(np.ceil(nTrainSamples/batchSize))

        inputIndexes = torch.randperm(nTrainSamples)
        X_train_scaled, T_train = X_train_scaled[inputIndexes], T_train[inputIndexes]

        for batchIdx in range(nBatches):
            batchStartIdx, batchStopIdx = batchIdx * batchSize, min(nTrainSamples, (batchIdx + 1) * batchSize)
            #if batchIdx == 0: print('epoch %d: starting batch %d out of %d' % (epochIdx, batchIdx, nBatches))
            data = X_train_scaled[batchStartIdx:batchStopIdx]#.cuda()
            label = T_train[batchStartIdx:batchStopIdx]#.cuda()

            optimizer.zero_grad()
            t_recon = model_PS(data)
            loss = loss_function_PS(t_recon[:, 0], label)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        trainLoss[epochIdx] = total_loss/nBatches
        lr_scheduler.step()

        model_PS.eval()
        t_recon = model_PS(X_test_scaled)
        loss = loss_function_PS(t_recon[:, 0], T_test)
        testLoss[epochIdx] = loss.item()

        if testLoss[epochIdx] < minTestLoss:
            minTestLoss = testLoss[epochIdx]
            torch.save(model_PS.state_dict(), './PS%d.pt' % datasetNum)

        nTotalDifferent = (t_recon[:, 0] - T_test).abs().sum()
        nTotal = T_test.numel()
        testProbLoss[epochIdx] = nTotalDifferent / nTotal

        t_recon = model_PS(X_train_scaled)
        nTotalDifferent = (t_recon[:, 0] - T_train).abs().sum()
        nTotal = T_train.numel()
        trainProbLoss[epochIdx] = nTotalDifferent / nTotal

        #print(f'epoch: {epochIdx}: trainLoss: {trainLoss[epochIdx]}; testLoss: {testLoss[epochIdx]}')

    plt.plot(trainLoss, label='train')
    plt.plot(testLoss, label='test')
    plt.plot(testProbLoss, label='testProbLoss')
    plt.plot(trainProbLoss, label='trainProbLoss')
    plt.legend()
    plt.xlabel('epoch')
    plt.title('loss at propensity-score train dataset%d' % datasetNum)
    plt.savefig('PS_loss_dataset%d' % datasetNum)
    #plt.show()
    plt.close()

    model_PS.load_state_dict(torch.load('./PS%d.pt' % datasetNum))
    model_PS.eval()

    # let's see the histogram of propensity score in the treated and control groups:
    X_scaled = scaler.transform(X)
    ps_scores = model_PS(torch.tensor(X_scaled, dtype=torch.float).cuda()).detach().cpu().numpy()
    X_scaled_treated, Y_treated = X_scaled[np.where(T)], Y[np.where(T)]
    X_scaled_control, Y_control = X_scaled[np.where(T==0)], Y[np.where(T==0)]
    PS_treated, PS_control = model_PS(torch.tensor(X_scaled_treated, dtype=torch.float).cuda()), model_PS(torch.tensor(X_scaled_control, dtype=torch.float).cuda())
    PS_treated, PS_control = PS_treated.detach().cpu().numpy()[:, 0], PS_control.detach().cpu().numpy()[:, 0]

    n_bins = 50
    n, bins, patches = plt.hist(Y_treated, n_bins, density=True, histtype='step', cumulative=False, label='treated')
    n, bins, patches = plt.hist(Y_control, n_bins, density=True, histtype='step', cumulative=False, label='control')
    plt.legend()
    plt.title('outcome (Y) values histogram %d' % datasetNum)
    plt.savefig('outcome_hist_dataset%d' % datasetNum)
    plt.close()
    #plt.show()

    n_bins = 50
    n, bins, patches = plt.hist(PS_treated, n_bins, density=True, histtype='step', cumulative=False, label='treated')
    n, bins, patches = plt.hist(PS_control, n_bins, density=True, histtype='step', cumulative=False, label='control')
    plt.legend()
    plt.title('Propensity scores histogram in dataset %d' % datasetNum)
    plt.savefig('PS_hist_dataset%d' % datasetNum)
    plt.close()
    #plt.show()

    X_scaled_treated, Y_treated = X_scaled_treated[np.where(PS_treated > lowPsTh)], Y_treated[np.where(PS_treated > lowPsTh)]
    X_scaled_control, Y_control = X_scaled_control[np.where(PS_control > lowPsTh)], Y_control[np.where(PS_control > lowPsTh)]
    PS_treated, PS_control = model_PS(torch.tensor(X_scaled_treated, dtype=torch.float).cuda()), model_PS(torch.tensor(X_scaled_control, dtype=torch.float).cuda())
    PS_treated, PS_control = PS_treated.detach().cpu().numpy()[:, 0], PS_control.detach().cpu().numpy()[:, 0]

    n_bins = 50
    n, bins, patches = plt.hist(PS_treated, n_bins, density=True, histtype='step', cumulative=False, label='treated')
    n, bins, patches = plt.hist(PS_control, n_bins, density=True, histtype='step', cumulative=False, label='control')
    plt.legend()
    plt.title('Propensity scores histogram in dataset %d post trimming' % datasetNum)
    plt.savefig('PS_hist_post_trimming_dataset%d' % datasetNum)
    plt.close()
    #plt.show()

    # calc IPW:
    meanOutcome_treated = Y_treated.mean()
    controlPsFactor = PS_control / (1-PS_control)
    meanOutcome_control = (Y_control * controlPsFactor).sum() / controlPsFactor.sum()

    mu_ATT_IPW = meanOutcome_treated - meanOutcome_control
    return mu_ATT_IPW, ps_scores

########################################## matching #######################################################
def calc_ATT_matching(datasetNum, scaler, X, T, Y, matchingMahalanobisThr):
    X_scaled = scaler.transform(X)
    X_scaled_treated, Y_treated = X_scaled[np.where(T)], Y[np.where(T)]
    X_scaled_control, Y_control = X_scaled[np.where(T == 0)], Y[np.where(T == 0)]

    #np.cov(m): Each row of m represents a variable, and each column a single observation of all those variables. Also see rowvar below.
    # calc the covariance matrix:
    X_cat = np.concatenate((X_scaled_treated, X_scaled_control), axis=0)
    S = np.cov(X_cat.transpose())
    S_inv = np.linalg.inv(S)

    nTreated, nControl = X_scaled_treated.shape[0], X_scaled_control.shape[0]
    treatedPairs = np.zeros((nTreated, 3))  # treatedIdx, coupleIdx in X_scaled_control and mahalanobisDist
    for treatedIdx in range(nTreated):
        #if treatedIdx % 100 == 0: print(f'started best mahalanobis calc for treated {treatedIdx} out of {nTreated}')
        treatedCovariates = np.expand_dims(X_scaled_treated[treatedIdx], axis=1)
        minMahalanobisDist = np.inf
        for controlIdx in range(nControl):
            controlCovariates = np.expand_dims(X_scaled_control[controlIdx], axis=1)
            d = treatedCovariates - controlCovariates
            mahalanobisDist = np.sqrt(np.matmul(d.transpose(), np.matmul(S_inv, d)))
            if mahalanobisDist < minMahalanobisDist:
                minMahalanobisDist = mahalanobisDist
                bestMatch = np.array([treatedIdx, controlIdx, mahalanobisDist])
        treatedPairs[treatedIdx] = bestMatch


    n_bins = 50
    n, bins, patches = plt.hist(treatedPairs[:, -1], n_bins, density=True, histtype='step', cumulative=True, label='mahalanobis')
    plt.legend()
    plt.grid(True)
    plt.title('Closest Mahalanobis distances CDF dataset%d' % datasetNum)
    plt.savefig('Closest_Mahalanobis_distances_CDF_dataset%d' % datasetNum)
    plt.close()
    #plt.show()

    treatedPairs = treatedPairs[np.where(treatedPairs[:, -1] < matchingMahalanobisThr)]
    treatedIndexes = treatedPairs[:, 0].astype('int')
    matchedControlIndexes = treatedPairs[:, 1].astype('int')
    mu_ATT_matching = Y_treated[treatedIndexes].mean() - Y_control[matchedControlIndexes].mean()
    return mu_ATT_matching

########################################## S-learner #######################################################
class sLearner(nn.Module):
    def __init__(self, covariateDim):
        super(sLearner, self).__init__()

        self.covariateDim = covariateDim
        self.internalDim = 1000#covariateDim
        # encoder:
        self.fc21 = nn.Linear(self.covariateDim, self.internalDim)
        self.fc22 = nn.Linear(self.internalDim, self.internalDim)
        self.fc23 = nn.Linear(self.internalDim, 1)
        #self.fc24 = nn.Linear(self.covariateDim, self.covariateDim)
        #self.fc25 = nn.Linear(self.covariateDim, 1)

        # general:
        self.logSoftMax = nn.LogSoftmax(dim=1)
        self.LeakyReLU = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x0):
        x1 = self.LeakyReLU(self.fc21(x0))
        x2 = self.LeakyReLU(self.fc22(x1))
        #x3 = self.fc23(x2)
        #x4 = self.LeakyReLU(self.fc24(x3))
        #x5 = self.fc25(x4)
        return self.fc23(x2)

    def forward(self, x):
        return self.encode(x)

def calc_ATT_sLearner(X_train_scaled, X_test_scaled, T_test, Y_test, T_train, Y_train, scaler, datasetNum, X, T):
    loss_sLearner = nn.L1Loss()
    model_S_learner = sLearner(covariateDim=X_train_scaled.shape[1] + 1).cuda()
    trainable_params = filter(lambda p: p.requires_grad, model_S_learner.parameters())
    config = json.load(open('my-config.json'))
    opt_name = config['optimizer']['type']
    opt_args = config['optimizer']['args']
    # opt_args['lr'] = 1e-4
    optimizer = getattr(torch.optim, opt_name)(trainable_params, **opt_args)
    # optimizer = optim.Adam(model_S_learner.parameters(), lr=1e-3)

    lr_name = config['lr_scheduler']['type']
    lr_args = config['lr_scheduler']['args']
    lr_args['step_size'] = 200
    if lr_name == 'None':
        lr_scheduler = None
    else:
        lr_scheduler = getattr(torch.optim.lr_scheduler, lr_name)(optimizer, **lr_args)

    nEpochs = 200 + 1
    trainLoss, trainProbLoss, testLoss, testProbLoss = np.zeros(nEpochs), np.zeros(nEpochs), np.zeros(nEpochs), np.zeros(nEpochs)
    X_test_scaled, T_test, Y_test = torch.tensor(X_test_scaled, dtype=torch.float).cuda(), torch.tensor(T_test, dtype=torch.float).unsqueeze_(-1).cuda(), torch.tensor(Y_test, dtype=torch.float).cuda()
    X_train_scaled, T_train, Y_train = torch.tensor(X_train_scaled, dtype=torch.float).cuda(), torch.tensor(T_train, dtype=torch.float).unsqueeze_(-1).cuda(), torch.tensor(Y_train, dtype=torch.float).cuda()
    nTrainSamples = X_train_scaled.shape[0]

    minTestLoss = np.inf
    for epochIdx in range(nEpochs):
        model_S_learner.train()
        total_loss = 0
        batchSize = 50
        nBatches = int(np.ceil(nTrainSamples / batchSize))

        inputIndexes = torch.randperm(nTrainSamples)
        X_train_scaled, T_train, Y_train = X_train_scaled[inputIndexes], T_train[inputIndexes], Y_train[inputIndexes]

        for batchIdx in range(nBatches):
            batchStartIdx, batchStopIdx = batchIdx * batchSize, min(nTrainSamples, (batchIdx + 1) * batchSize)
            # if batchIdx == 0: print('epoch %d: starting batch %d out of %d' % (epochIdx, batchIdx, nBatches))
            data = torch.cat((X_train_scaled[batchStartIdx:batchStopIdx], T_train[batchStartIdx:batchStopIdx]), dim=1)
            label = Y_train[batchStartIdx:batchStopIdx]  # .cuda()

            optimizer.zero_grad()
            t_recon = model_S_learner(data)
            loss = loss_sLearner(t_recon[:, 0], label)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        trainLoss[epochIdx] = total_loss / nBatches
        lr_scheduler.step()

        model_S_learner.eval()
        t_recon = model_S_learner(torch.cat((X_test_scaled, T_test), dim=1))
        loss = loss_sLearner(t_recon[:, 0], Y_test)
        testLoss[epochIdx] = loss.item()

        if testLoss[epochIdx] < minTestLoss:
            minTestLoss = testLoss[epochIdx]
            torch.save(model_S_learner.state_dict(), 'S_learner%d.pt' % datasetNum)

        # print(f'epoch: {epochIdx}: trainLoss: {trainLoss[epochIdx]}; testLoss: {testLoss[epochIdx]}')

    n_bins = 50
    n, bins, patches = plt.hist(t_recon.detach().cpu().numpy(), n_bins, density=True, histtype='step', cumulative=False, label='est_outcome')
    n, bins, patches = plt.hist(Y_test.detach().cpu().numpy(), n_bins, density=True, histtype='step', cumulative=False, label='true_outcome')
    plt.legend()
    plt.grid(True)
    plt.title('S-learner: True and est outcome hist dataset%d' % datasetNum)
    plt.savefig('Slearner_True_and_est_outcome_hist_dataset%d' % datasetNum)
    plt.close()
    #plt.show()

    outcomeStd = Y_train.cpu().numpy().std()
    plt.plot(trainLoss / outcomeStd, label='train')
    plt.plot(testLoss / outcomeStd, label='test')
    # plt.plot(testProbLoss, label='testProbLoss')
    # plt.plot(trainProbLoss, label='trainProbLoss')
    plt.legend()
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('outcome std''s')
    plt.title('S-learner: L1-loss train dataset%d; std(Y)=%2.2f' % (datasetNum, outcomeStd))
    plt.savefig('Slearner_L1loss_train_dataset%d' % datasetNum)
    plt.close()
    # plt.show()

    model_S_learner.load_state_dict(torch.load('S_learner%d.pt' % datasetNum))
    model_S_learner.eval()
    X_scaled = scaler.transform(X)
    X_scaled_treated = X_scaled[np.where(T)]

    treatmentEstOutcome = model_S_learner(torch.cat((torch.tensor(X_scaled_treated, dtype=torch.float), torch.ones(X_scaled_treated.shape[0], 1)), dim=1).cuda())
    controlEstOutcome = model_S_learner(torch.cat((torch.tensor(X_scaled_treated, dtype=torch.float), torch.zeros(X_scaled_treated.shape[0], 1)), dim=1).cuda())

    mu_ATT_sLearner = treatmentEstOutcome.detach().cpu().numpy().mean() - controlEstOutcome.detach().cpu().numpy().mean()

    return mu_ATT_sLearner

########################################## T-learner #######################################################
def calc_ATT_tLearner(scaler, X_train_scaled, T_train, T_test, X_test_scaled, Y_test, Y_train, datasetNum, X, T):
    loss_tLearner = nn.L1Loss()
    # train treated model
    model_T_learner_treated = sLearner(covariateDim=X_train_scaled.shape[1]).cuda()
    trainable_params = filter(lambda p: p.requires_grad, model_T_learner_treated.parameters())
    config = json.load(open('my-config.json'))
    opt_name = config['optimizer']['type']
    opt_args = config['optimizer']['args']
    # opt_args['lr'] = 1e-4
    optimizer = getattr(torch.optim, opt_name)(trainable_params, **opt_args)
    # optimizer = optim.Adam(model_S_learner.parameters(), lr=1e-3)

    lr_name = config['lr_scheduler']['type']
    lr_args = config['lr_scheduler']['args']
    lr_args['step_size'] = 200
    if lr_name == 'None':
        lr_scheduler = None
    else:
        lr_scheduler = getattr(torch.optim.lr_scheduler, lr_name)(optimizer, **lr_args)

    nEpochs = 200 + 1
    trainLoss, trainProbLoss, testLoss, testProbLoss = np.zeros(nEpochs), np.zeros(nEpochs), np.zeros(
        nEpochs), np.zeros(nEpochs)

    treatedIndexes_train, controlIndexes_train = np.where(T_train), np.where(T_train == 0)
    treatedIndexes_test, controlIndexes_test = np.where(T_test), np.where(T_test == 0)

    X_test_scaled_treated, T_test_treated, Y_test_treated = torch.tensor(X_test_scaled[treatedIndexes_test], dtype=torch.float).cuda(), torch.tensor(T_test[treatedIndexes_test], dtype=torch.float).unsqueeze_(-1).cuda(), torch.tensor(Y_test[treatedIndexes_test], dtype=torch.float).cuda()
    X_train_scaled_treated, T_train_treated, Y_train_treated = torch.tensor(X_train_scaled[treatedIndexes_train], dtype=torch.float).cuda(), torch.tensor(T_train[treatedIndexes_train], dtype=torch.float).unsqueeze_(-1).cuda(), torch.tensor(Y_train[treatedIndexes_train], dtype=torch.float).cuda()
    minTestLoss = np.inf
    for epochIdx in range(nEpochs):
        model_T_learner_treated.train()
        total_loss = 0
        batchSize = 50
        nTrainSamples_treated = X_train_scaled_treated.shape[0]
        nBatches = int(np.ceil(nTrainSamples_treated / batchSize))

        inputIndexes = torch.randperm(nTrainSamples_treated)
        X_train_scaled_treated, T_train_treated, Y_train_treated = X_train_scaled_treated[inputIndexes], T_train_treated[inputIndexes], Y_train_treated[inputIndexes]

        for batchIdx in range(nBatches):
            batchStartIdx, batchStopIdx = batchIdx * batchSize, min(nTrainSamples_treated, (batchIdx + 1) * batchSize)
            # if batchIdx == 0: print('epoch %d: starting batch %d out of %d' % (epochIdx, batchIdx, nBatches))
            data = X_train_scaled_treated[batchStartIdx:batchStopIdx]
            label = Y_train_treated[batchStartIdx:batchStopIdx]  # .cuda()

            optimizer.zero_grad()
            t_recon = model_T_learner_treated(data)
            loss = loss_tLearner(t_recon[:, 0], label)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        trainLoss[epochIdx] = total_loss / nBatches
        lr_scheduler.step()

        model_T_learner_treated.eval()
        t_recon = model_T_learner_treated(X_test_scaled_treated)
        loss = loss_tLearner(t_recon[:, 0], Y_test_treated)
        testLoss[epochIdx] = loss.item()

        if testLoss[epochIdx] < minTestLoss:
            minTestLoss = testLoss[epochIdx]
            torch.save(model_T_learner_treated.state_dict(), 'T_learner_treated%d.pt' % datasetNum)

        # print(f'epoch: {epochIdx}: trainLoss: {trainLoss[epochIdx]}; testLoss: {testLoss[epochIdx]}')

    n_bins = 50
    n, bins, patches = plt.hist(t_recon.detach().cpu().numpy(), n_bins, density=True, histtype='step', cumulative=False, label='est_outcome')
    n, bins, patches = plt.hist(Y_test_treated.detach().cpu().numpy(), n_bins, density=True, histtype='step', cumulative=False, label='true_outcome')
    plt.legend()
    plt.grid(True)
    plt.title('T-learner: True and est outcome for treated hist dataset%d' % datasetNum)
    plt.savefig('tlearner_True_and_est_outcome_hist_treated_dataset%d' % datasetNum)
    plt.close()
    #plt.show()

    outcomeStd = Y_train.std()
    plt.plot(trainLoss / outcomeStd, label='train')
    plt.plot(testLoss / outcomeStd, label='test')
    # plt.plot(testProbLoss, label='testProbLoss')
    # plt.plot(trainProbLoss, label='trainProbLoss')
    plt.legend()
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('outcome std''s')
    plt.title('T-learner: L1-loss train treated dataset%d; std(Y)=%2.2f' % (datasetNum, outcomeStd))
    plt.savefig('tlearner_L1loss_train_treated_dataset%d' % datasetNum)
    plt.close()
    #plt.show()

    # train control model
    model_T_learner_control = sLearner(covariateDim=X_train_scaled.shape[1]).cuda()
    trainable_params = filter(lambda p: p.requires_grad, model_T_learner_control.parameters())
    config = json.load(open('my-config.json'))
    opt_name = config['optimizer']['type']
    opt_args = config['optimizer']['args']
    # opt_args['lr'] = 1e-4
    optimizer = getattr(torch.optim, opt_name)(trainable_params, **opt_args)
    # optimizer = optim.Adam(model_S_learner.parameters(), lr=1e-3)

    lr_name = config['lr_scheduler']['type']
    lr_args = config['lr_scheduler']['args']
    lr_args['step_size'] = 200
    if lr_name == 'None':
        lr_scheduler = None
    else:
        lr_scheduler = getattr(torch.optim.lr_scheduler, lr_name)(optimizer, **lr_args)

    nEpochs = 200 + 1
    trainLoss, trainProbLoss, testLoss, testProbLoss = np.zeros(nEpochs), np.zeros(nEpochs), np.zeros(nEpochs), np.zeros(nEpochs)

    X_test_scaled_control, T_test_control, Y_test_control = torch.tensor(X_test_scaled[controlIndexes_test], dtype=torch.float).cuda(), torch.tensor(T_test[controlIndexes_test], dtype=torch.float).unsqueeze_(-1).cuda(), torch.tensor(Y_test[controlIndexes_test], dtype=torch.float).cuda()
    X_train_scaled_control, T_train_control, Y_train_control = torch.tensor(X_train_scaled[controlIndexes_train], dtype=torch.float).cuda(), torch.tensor(T_train[controlIndexes_train], dtype=torch.float).unsqueeze_(-1).cuda(), torch.tensor(Y_train[controlIndexes_train], dtype=torch.float).cuda()

    minTestLoss = np.inf
    for epochIdx in range(nEpochs):
        model_T_learner_control.train()
        total_loss = 0
        batchSize = 50
        nTrainSamples_control = X_train_scaled_control.shape[0]
        nBatches = int(np.ceil(nTrainSamples_control / batchSize))

        inputIndexes = torch.randperm(nTrainSamples_control)
        X_train_scaled_control, T_train_control, Y_train_control = X_train_scaled_control[inputIndexes], T_train_control[inputIndexes], Y_train_control[inputIndexes]

        for batchIdx in range(nBatches):
            batchStartIdx, batchStopIdx = batchIdx * batchSize, min(nTrainSamples_control, (batchIdx + 1) * batchSize)
            # if batchIdx == 0: print('epoch %d: starting batch %d out of %d' % (epochIdx, batchIdx, nBatches))
            data = X_train_scaled_control[batchStartIdx:batchStopIdx]
            label = Y_train_control[batchStartIdx:batchStopIdx]  # .cuda()

            optimizer.zero_grad()
            t_recon = model_T_learner_control(data)
            loss = loss_tLearner(t_recon[:, 0], label)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        trainLoss[epochIdx] = total_loss / nBatches
        lr_scheduler.step()

        model_T_learner_control.eval()
        t_recon = model_T_learner_control(X_test_scaled_control)
        loss = loss_tLearner(t_recon[:, 0], Y_test_control)
        testLoss[epochIdx] = loss.item()

        if testLoss[epochIdx] < minTestLoss:
            minTestLoss = testLoss[epochIdx]
            torch.save(model_T_learner_control.state_dict(), 'T_learner_control%d.pt' % datasetNum)

        # print(f'epoch: {epochIdx}: trainLoss: {trainLoss[epochIdx]}; testLoss: {testLoss[epochIdx]}')

    n_bins = 50
    n, bins, patches = plt.hist(t_recon.detach().cpu().numpy(), n_bins, density=True, histtype='step', cumulative=False, label='est_outcome')
    n, bins, patches = plt.hist(Y_test_control.detach().cpu().numpy(), n_bins, density=True, histtype='step', cumulative=False, label='true_outcome')
    plt.legend()
    plt.grid(True)
    plt.title('T-learner: True and est outcome for control hist dataset%d' % datasetNum)
    plt.savefig('tlearner_True_and_est_outcome_hist_control_dataset%d' % datasetNum)
    plt.close()
    # plt.show()

    outcomeStd = Y_train.std()
    plt.plot(trainLoss / outcomeStd, label='train')
    plt.plot(testLoss / outcomeStd, label='test')
    # plt.plot(testProbLoss, label='testProbLoss')
    # plt.plot(trainProbLoss, label='trainProbLoss')
    plt.legend()
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('outcome std''s')
    plt.title('T-learner: L1-loss control dataset%d; std(Y)=%2.2f' % (datasetNum, outcomeStd))
    plt.savefig('tlearner_L1loss_train_control_dataset%d' % datasetNum)
    plt.close()
    # plt.show()

    model_T_learner_treated.load_state_dict(torch.load('T_learner_treated%d.pt' % datasetNum))
    model_T_learner_treated.eval()
    model_T_learner_control.load_state_dict(torch.load('T_learner_control%d.pt' % datasetNum))
    model_T_learner_control.eval()

    X_scaled = scaler.transform(X)
    X_scaled_treated = X_scaled[np.where(T)]
    X_scaled_control = X_scaled[np.where(T == 0)]

    treatmentEstOutcome = model_T_learner_treated(torch.tensor(X_scaled_treated, dtype=torch.float).cuda())
    controlEstOutcome = model_T_learner_control(torch.tensor(X_scaled_treated, dtype=torch.float).cuda())

    mu_ATT_tLearner = treatmentEstOutcome.detach().cpu().numpy().mean() - controlEstOutcome.detach().cpu().numpy().mean()
    return mu_ATT_tLearner