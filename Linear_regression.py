# gradient > 0 -> left / gradient < 0 -> right
# w = w - alpha(d(loss)/dw) = w - alpha*2x(xw-y)  at loss = (xw-y)^2
# alpha = learning rate
# loss를 구한 후, d(loss)/dw 구할 때 back_propagation 이용하면 빨리 구할 수 있다
# -> using chain rule
# d(loss)/dw = w.grad.data -> 이게 0이 되어야 좋은

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))

# first define model
class Model(torch.nn.Module):
    def __init__(self):

        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # (input size, outputsize)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = Model()

# Second, Construct loss and optimizer
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # what parameter to update


# Third, Training

for epoch in range(500):
    # 한꺼번에 계산하는 방, one by one은 for문으로 하나하나 계산하는
    y_pred = model(x_data)

    loss = criterion(y_pred, y_data)  # forward
    print(epoch, loss.data.item())

    optimizer.zero_grad()  # loss 계산하기 전에 loss 저장된 buffer 초기화하는 기능
    loss.backward()  # backword
    optimizer.step()  # to update variable


var = Variable(torch.Tensor([4.0]))
print('predict: ', 4, model.forward(var).data.item())