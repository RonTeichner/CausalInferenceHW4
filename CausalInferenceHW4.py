import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import json

# Read the data and create train and test sets:
data = pd.read_csv('data1.csv', header = None)
charColumns = np.array([2, 21, 24])
floatColumns = np.setdiff1d(np.arange(1, 61), charColumns)

dataMat = np.zeros((data.values.shape[0]-1, data.values.shape[1]-1), dtype='float')
for colIdx in floatColumns:
    dataMat[:, colIdx-1] = np.array(data.values[1:, colIdx], dtype='float')

for colIdx in charColumns:
    charColumn = data.values[1:, colIdx]
    uniqueValues = np.unique(charColumn)
    for uniqueValIdx, uniqueVal in enumerate(uniqueValues):
        dataMat[np.where(charColumn == uniqueVal), colIdx-1] = uniqueValIdx

X, T, Y = dataMat[:, :-2], dataMat[:, -2], dataMat[:, -1]

nSamples = X.shape[0]
nTrainSamples = int(np.round(0.8*nSamples))
trainIndexes = np.sort(np.random.permutation(nSamples)[:nTrainSamples])
testIndexes = np.setdiff1d(np.arange(1, nSamples), trainIndexes)

X_train, T_train, Y_train = X[trainIndexes, :], T[trainIndexes], Y[trainIndexes]
X_test, T_test, Y_test = X[testIndexes, :], T[testIndexes], Y[testIndexes]

# scale each covariate:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# remove the tails outside 3std's
goodIndexes = np.arange(X_train.shape[0])
maxStd = 5
for featureIdx in range(X_train_scaled.shape[1]):
    featureValues = X_train_scaled[:, featureIdx]
    goodFeatureIndexes = np.intersect1d(np.where(-maxStd < featureValues), np.where(featureValues < maxStd))
    goodIndexes = np.intersect1d(goodIndexes, goodFeatureIndexes)

print(f'out of {X_train.shape[0]} samples we remained with {goodIndexes.shape[0]} after removing outlayers')
nTrainSamples = goodIndexes.shape[0]
X_train, T_train, Y_train = X_train[goodIndexes, :], T_train[goodIndexes], Y_train[goodIndexes]

# scale each covariate:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled, T_train = torch.tensor(X_train_scaled, dtype=torch.float), torch.tensor(T_train, dtype=torch.float)
X_test_scaled, T_test = torch.tensor(X_test_scaled, dtype=torch.float), torch.tensor(T_test, dtype=torch.float)

# estimate the propensity score:
class propensityModel(nn.Module):
    def __init__(self, covariateDim):
        super(propensityModel, self).__init__()

        self.covariateDim = covariateDim
        # encoder:
        self.fc21 = nn.Linear(self.covariateDim, self.covariateDim)
        self.fc22 = nn.Linear(self.covariateDim, 1)

        # general:
        self.logSoftMax = nn.LogSoftmax(dim=1)
        self.LeakyReLU = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x0):
        x1 = self.LeakyReLU(self.fc21(x0))
        x2 = self.fc22(x1)
        return x2

    def forward(self, x):
        return self.sigmoid(self.encode(x))

def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x)
    return BCE

model = propensityModel(covariateDim=X_train_scaled.shape[1]).cuda()

trainable_params = filter(lambda p: p.requires_grad, model.parameters())

config = json.load(open('my-config.json'))
config['net_mode'] = 'init'
config['cfg'] = 'crnn.cfg'

opt_name = config['optimizer']['type']
opt_args = config['optimizer']['args']
#opt_args['lr'] = 1e-3
optimizer = getattr(torch.optim, opt_name)(trainable_params, **opt_args)
#optimizer = optim.Adam(model.parameters(), lr=1e-3)

lr_name = config['lr_scheduler']['type']
lr_args = config['lr_scheduler']['args']
#lr_args['step_size'] = 1000
if lr_name == 'None':
    lr_scheduler = None
else:
    lr_scheduler = getattr(torch.optim.lr_scheduler, lr_name)(optimizer, **lr_args)

nEpochs = 1000+1
trainLoss, testLoss = np.zeros(nEpochs), np.zeros(nEpochs)

for epochIdx in range(nEpochs):
    model.train()
    total_loss = 0
    batchSize = 100
    nBatches = int(np.ceil(nTrainSamples/batchSize))

    inputIndexes = torch.randperm(nTrainSamples)
    X_train_scaled, T_train = X_train_scaled[inputIndexes], T_train[inputIndexes]

    for batchIdx in range(nBatches):
        batchStartIdx, batchStopIdx = batchIdx * batchSize, min(nTrainSamples, (batchIdx + 1) * batchSize)
        if batchIdx == 0: print('epoch %d: starting batch %d out of %d' % (epochIdx, batchIdx, nBatches))
        data = X_train_scaled[batchStartIdx:batchStopIdx].cuda()
        label = T_train[batchStartIdx:batchStopIdx].cuda()

        optimizer.zero_grad()
        t_recon = model(data)
        loss = loss_function(t_recon, label)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    trainLoss[epochIdx] = total_loss/nBatches
    lr_scheduler.step()

    model.eval()
    t_recon = model(X_test_scaled.cuda())
    loss = loss_function(t_recon, T_test.cuda())
    testLoss[epochIdx] = loss.item()

    print(f'epoch: {epochIdx}: trainLoss: {trainLoss[epochIdx]}; testLoss: {testLoss[epochIdx]}')
