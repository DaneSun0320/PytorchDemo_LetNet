#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @IDE          :PyCharm
# @Project      :PytorchDemo_LetNet
# @FileName     :model
# @CreatTime    :2022/5/9 10:28 
# @CreatUser    :DaneSun

import torch.nn as nn
import torch.nn.functional as F

class LetNet(nn.Module):
    # 构造函数定义网络结构
    def __init__(self):
        # 调用父类构造函数
        super(LetNet, self).__init__()
        # 定义卷积层1 输入3通道，输出16通道，卷积核大小5*5
        self.conv1 = nn.Conv2d(3,16,5) # input:(3,32,32) output:(16,28,28)
        # 定义池化层1
        self.pool1 = nn.MaxPool2d(2,2) # input:(16,28,28) output:(16,14,14)
        # 定义卷积层2 输入16通道，输出32通道，卷积核大小5*5
        self.conv2 = nn.Conv2d(16,32,5) # input:(16,14,14) output:(32,10,10)
        # 定义池化层2
        self.pool2 = nn.MaxPool2d(2,2) # input:(32,10,10) output:(32,5,5)
        # 定义全连接层1
        self.fc1 = nn.Linear(32*5*5,120) # input:(32*5*5) output:(120)
        # 定义全连接层2
        self.fc2 = nn.Linear(120,84) # input:(120) output:(84)
        # 定义全连接层3
        self.fc3 = nn.Linear(84,10) # input:(84) output:(10) (10个分类)

    def forward(self, x):
        # 定义卷积层1的前向传播过程
        x = F.relu(self.conv1(x))
        # 定义池化层1的前向传播过程
        x = self.pool1(x)
        # 定义卷积层2的前向传播过程
        x = F.relu(self.conv2(x))
        # 定义池化层2的前向传播过程
        x = self.pool2(x)
        # 定义全连接层1的前向传播过程
        x = x.view(-1, 32*5*5) # 将x变成一维向量
        x = F.relu(self.fc1(x))
        # 定义全连接层2的前向传播过程
        x = F.relu(self.fc2(x))
        # 定义全连接层3的前向传播过程
        x = self.fc3(x)
        return x

# 测试网络
if __name__ == '__main__':
    import torch
    input1 = torch.randn([32,3,32,32])
    model = LetNet()
    print(model)
    outpput = model(input1)