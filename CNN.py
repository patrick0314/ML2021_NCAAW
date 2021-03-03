import os
import numpy as np
import pandas as pd
import torch
import torchvision.dataset
import torchvision.transforms
import torch.nn.init

from ReadData import *

# Data Preprocess
print('=== data preprocess ===')
pd.options.mode.chained_assignment = None # default = 'warn'

team = Team() # [369, 2] = [TeamID, TeamName]
n = len(team['TeamID'])
VS = np.zeros((n, n)) # [369, 369]

compactresult = CompactResult() # [1386, 8] = [Season, DayNum, WteamID, LteamID, LScore, WLoc, NumOT]
for i in range(len(compactresult['Season'])):
    W = team.index[team['TeamID']==compactresult['WTeamID'][i]].to_list()
    L = team.index[team['TeamID']==compactresult['LTeamID'][i]].to_list()
    VS[W[0], L[0]] += 1

regularcompactresult = RegularCompactResult() # [112183, 8] = [Season, DayNum, WteamID, WScore, LteamID, LScore, WLoc, NumOT]
for i in range(len(regularcompactresult['Season'])):
    W = team.index[team['TeamID']==regularcompactresult['WTeamID'][i]].to_list()
    L = team.index[team['TeamID']==regularcompactresult['LTeamID'][i]].to_list()
    VS[W[0], L[0]] += 1

for i in range(VS.shape[0]):
    for j in range(VS.shape[1]):
        tmp = VS[i, j] + VS[j, i]
        if tmp != 0:
            VS[i, j] /= tmp

sample = Output()
pair = []
for i in range(len(output['ID'])):
    pair.append(output['ID'][i].split('_')[1], output['ID'][i].split('_')[2])

label = []
for i in pair:
    label.append(VS[i[0], i[1]])

# Data Set
print('=== data set ===')
class Dataset(torch.utils.data.Dataset):
    def __init__(self, pair, VS, mode='train'):
        self.pair = pair

    def __len__(self):
        return len(self.pair)

    def __getitem(self, pair, label):
        
        return (, y)

training = Dataset(pair, label)

# DataSet Loader
print('=== dataset loader ===')
data_loader = torch.utils.data.DataLoader(
        dataset=training, 
        batch_size=100,
        shuffle=True,
        drop_last=True
        )

# CNN Model
print('=== model build ===')
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential()
        self.layer2 = torch.nn.Sequential()
        self.fc = torch.nn.Linear(, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# parameters
print('=== parameter setting ===')
learning_rate = 0.001
tranining_epochs = 15

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN().to(device)

# define cost/loss & optimizer
loss = torch.nn.CrossEntropyLoss().to(device) # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train my model
print('=== laerning start ===')
total_batch = len(data_loader)
for epoch in range(training_epochs):
    avg_cost = 0

    for x, y in enumerate(data_loader):
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

print('Learning Finished!')
