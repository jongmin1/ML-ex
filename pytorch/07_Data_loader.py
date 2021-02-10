# for multi-input
# x -> ... (linear,activation) ... -> y
# data size 커지면 data loader 사용해야 함(batch같은 거 다 해줌)
# 학습을 위한 방대한 데이터를 미니배치 단위로 처리할 수 있고,
# 데이터를 무작위로 섞음으로써 학습의 효율성을 향상시킬 수 있다

import torch.nn.functional as F

import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# data => csv file(col별로 값이 index)로


class getData(Dataset):
    def __init__(self):
        xy = np.loadtxt('data.csv', delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = Variable(torch.from_numpy(xy[:, 0:-1]))
        self.y_data = Variable(torch.from_numpy(xy[:, [-1]]))

    def __getitem__(self, index):
        return self.x_Data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = getData()
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,  # when train
                          num_workers=2  # in multiple processor
                          )


# define model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(8, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred


model = Model()

# Condtruct loss and optimizer
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Training
for epoch in range(100):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        inputs, labels = Variable(inputs), Variable(labels)

        y_pred = model(inputs)

        loss = criterion(y_pred, labels)
        print(epoch, i, loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

