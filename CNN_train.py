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
result = np.concatenate((result, np.ones((result.shape[0], 2))), axis=1) # add bias [170992, 5]

for i in range(len(result)):
    if int(result[i, 1]) < int(result[i, 2]):
        continue
    else:
        result[i, 1], result[i, 2] = result[i, 2], result[i, 1]
        result[i, 4] = 0

lable = result[:, 4] # [170992]
result = np.delete(result, 4, axis=1)

x = result
print('x: ', np.shape(x))
y = lable
print('y: ', np.shape(y))

# dataset dataloader
x = torch.from_numpy(x)
y = torch.from_numpy(y)
dataset = Data.TensorDataset(x, y)
dataloader = Data.DataLoader(dataset, batch_size=256)

# model build
class CNN(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn = nn.Sequential(

                )
        self.fc = nn.Sequential(

                )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

# parameter setting
learning_rate = 0.1
epochs = 1000
best = 0.0

model = CNN().cuda()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training
print('=== start training ===')
for epoch in range(epochs):
    acc = 0.0

    model.train()
    for i, (x, y) in enumerate(dataloader):
        pred = model(x.cuda())
        batch_loss = loss(pred, y.cuda())
        batch_loss.backward()
        optimizer.step()

        acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == y.numpy())

    acc = acc / len(dataloader)
    if acc > best:
        torch.save(model.state_dict(), 'model/model.pth')
        best = acc
        print('model save')
