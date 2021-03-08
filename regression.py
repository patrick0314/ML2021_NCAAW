import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import torch.nn as nn

from ReadData import *

# Data Preprocess
print('=== data pre-process ===')
compactresult = CompactResult() # [1386, 8]
detailedresult = DetailedResult() # [630, 34] = [Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc ... LOR, LDR, LAst, LTO, LStl, LBlk, LPF]
regularcompactresult = RegularCompactResult() # [112183, 8]
regulardetailedresult = RegularDetailedResult() # [56793, 34]] = [Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc ... LOR, LDR, LAst, LTO, LStl, LBlk, LPF]
tmp = pd.concat([compactresult, regularcompactresult], axis=0, ignore_index=True)
tmp = tmp.drop(['DayNum', 'WScore', 'LScore', 'NumOT', 'WLoc'], axis=1)
tmp1 = pd.concat([detailedresult, regulardetailedresult], axis=0, ignore_index=True) # [57423, 34]
tmp1 = tmp1.drop(['DayNum', 'WScore', 'LScore', 'NumOT', 'WLoc', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF'], axis=1)
tmp1 = tmp1.drop(['LFGM', 'LFGA',  'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF'], axis=1)
result = pd.concat([tmp, tmp1], axis=0, ignore_index=True)
result = result.to_numpy()
result = result.astype(float) # [170992, 3]
result = np.concatenate((result, np.ones((result.shape[0], 1))), axis=1) # add bias [170992, 4]

lable = np.zeros(len(result)) # [170992, 1]
for i in range(len(result)):
    if int(result[i, 1] ) < int(result[i, 2]):
        lable[i] = 1
    else:
        lable[i] = 0

# 
x = result
y = lable

# parameter
print('=== parameter setting ===')
w = np.zeros(len(x[0]))
lr = 1
iteration = 1000
s_grad = np.zeros(len(x[0]))
x_t = x.transpose()

# linear regression training
print('=== start regression ===')
for i in range(iteration):
    if i % 100 == 99:
        print('iteration: ', i+1, '  ', w)
    pred = np.dot(x, w)
    loss = pred - y
    grad = 2 * np.dot(x_t, loss)
    s_grad += grad**2
    ada = np.sqrt(s_grad)
    w = w - lr * grad / ada

print('=== regression complete ===')

# test
print('=== start prediction ===')
output = Output()
output = output
output = output.to_numpy()

for i in range(len(output)):
    S = int(output[i][0][0:4])
    W = int(output[i][0][5:9])
    L = int(output[i][0][10:])
    x = np.array([S, W, L, 1.0])
    pred = np.dot(x, w)
    output[i][1] = pred

# save file
print('=== save file ===')
df = pd.DataFrame(output, columns=['ID', 'Pred'])
df.to_csv('regression.csv', index=0)
