# gradient > 0 -> left / gradient < 0 -> right
# w = w - alpha(d(loss)/dw) = w - alpha*2x(xw-y)  at loss = (xw-y)^2
# alpha = learning rate
# d(loss)/dw 구할 때 back_propagation 이용하면 빨리 구할 수 있다 -> using chain rule
# d(loss)/dw = w.grad.data

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = Variable(torch.Tensor([1.0]), requires_grad=True)


def forward(x):
    return x * w


def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


def gradient(x,y):
    return 2*x*(x*w - y)


for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward()
        print('\tgrad:', x_val, y_val, w.grad.data[0])
        w.data = w.data - 0.01 * w.grad.data

        w.grad.data.zero_()

    print('progress:', epoch, l.data[0])

print('predict', '4 ->', forward(4).data[0])











