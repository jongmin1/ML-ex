# look at small portion of image at a once
# filter(각 weight이 적힌) 씌워서 줄여나가
# patch
# pooling -> max pooling이라면 가장 큰 수 뽑아서 대표로 사용(크기 줄일 때)
# padding -> 원래 크기를 유지시켜주기
# stride => 움직이는  / 한 칸 식 움직이면 no stride

from torch import nn
import numpy as np
import torch.nn.functional as F


class Net (nn.Moduele):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)

        # 320 그냥 돌리고 에러뜨는거 보고 수 맞춰주면 됨
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return F.log_softmax(x)