# gradient > 0 -> left / gradient < 0 -> right
# w = w - alpha(d(loss)/dw) = w - alpha*2x(xw-y)  at loss = (xw-y)^2
# alpha = learning rate

import numpy as np
import matplotlib.pyplot as plt
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


def forward(x):
    return x*w


w_list = []
mse_list = []


def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


def gradient(x,y):
    return 2*x*(x*w - y)


for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        w = w - 0.01*grad
        print('\tgrad:', x_val, y_val, grad)
        l = loss(x_val, y_val)

    print('progress:', epoch, 'w=', w, 'loss=', l)

print('predict', '4 ->', forward(4))


