#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @IDE          :PyCharm
# @Project      :PytorchDemo_LetNet
# @FileName     :predict
# @CreatTime    :2022/5/9 13:19 
# @CreatUser    :DaneSun

import torch
from PIL import Image
from torchvision import transforms

from model import LetNet

transform = transforms.Compose([
    transforms.Resize((32, 32)), # 将图片转换为32*32
    transforms.ToTensor(), # 将图片转换为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = LetNet()
net.load_state_dict(torch.load('./model/letnet.pth'))

img = Image.open('test.jpg') # 加载图片 [height, width, channel]
img = transform(img) # 将图片转换成tensor [channel, height, width]
img = torch.unsqueeze(img, 0) # 增加一个维度 [batch, channel, height, width]

with torch.no_grad():
    output = net(img)
    predict = torch.max(output, 1)[1].data.numpy()
    # torch.softmax(output, 1)  计算概率
print(classes[int(predict)])