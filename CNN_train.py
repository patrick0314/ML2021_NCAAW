import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import torch.nn as nn

from ReadData import *

# Data Preprocess

detailedresult = DetailedResult() # [630, 34] = [Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc ... LOR, LDR, LAst, LTO, LStl, LBlk, LPF]
regulardetailedresult = RegularDetailedResult() # [56793, 34]] = [Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc ... LOR, LDR, LAst, LTO, LStl, LBlk, LPF]
detail = pd.concat([detailedresult, regulardetailedresult], axis=0, ignore_index=True) # [57423, 34]
detail = detail.drop(['WLoc'], axis=1)
detail = detail.to_numpy()
detail = detail.astype(float)

lable = []
for i in range(len(detail)):
    if int(detail[i, 2] ) < int(detail[i, 4]):
        lable.append(1)
    else:
        lable.append(0)

detail = torch.from_numpy(detail)
lable  = torch.tensor(lable)

# Data Set
print('=== data set ===')
dataset = Data.TensorDataset(detail, lable)

# DataSet Loader
print('=== dataset loader ===')
dataloader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=100,
        shuffle=True,
        )

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(33, 16),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(16, 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.fc(x)

# parameters
print('=== parameter setting ===')
learning_rate = 0.0001
training_epochs = 15

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Classifier().to(device)

# define cost/loss & optimizer
loss = nn.CrossEntropyLoss().to(device) # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train my model
print('=== laerning start ===')
total_batch = len(dataloader) # 575
best_acc = 0.0
for epoch in range(training_epochs):
    print('epoch: ', epoch, end='   ')
    model.train()
    avg_cost = 0
    for step, (x, y) in enumerate(dataloader):
        # label is not one-hot encoded
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        train_pred = model(x)
        batch_loss = loss(train_pred, y)
        batch_loss.bachward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1)==y.numpy())
    
    train_acc /= len(dataloader)
    if train_acc > best_acc:
        torch.save(model.state_dict(), 'model/CNN_train.pth')
        print('model save !!!!!')
        best_acc = train_acc

print('=== learning complete ===')
