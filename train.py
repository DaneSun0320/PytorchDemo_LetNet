#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @IDE          :PyCharm
# @Project      :PytorchDemo_LetNet
# @FileName     :train
# @CreatTime    :2022/5/9 11:12 
# @CreatUser    :DaneSun
import os

import torch
from torch import nn, optim
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from model import LetNet
from utils import img_show

# 定义超参数
EPOCH = 50
BATCH_SIZE = 36

# 判断GPU是否可用
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE:', DEVICE)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# 下载训练集 首次运行需要将download设置为True
train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

# 加载训练集 batch_size每次训练的图片数 shuffle打乱数据集 num_workers子线程数
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# 导入测试集
test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

# 加载测试集
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10000, shuffle=False, num_workers=0)

# 将测试集加载器转换为迭代器
test_iter = iter(test_loader)
test_image, test_label = test_iter.next()
test_image, test_label = test_image.to(DEVICE), test_label.to(DEVICE)

# 定义标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 查看标签
# print(' '.join('%5s' % classes[test_label[j]] for j in range(5)))
# 查看图片
# img_show(make_grid(test_image))


net = LetNet().to(DEVICE)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001) # lr学习率

for epoch in range(EPOCH):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        # 获取输入数据 和标签
        inputs, labels = data

        # 将数据加载到DEVICE
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # 将优化器梯度置0
        optimizer.zero_grad()

        # 前向传播 获取预测值 和 损失 (预测值与标签进行比较) 反向传播 计算梯度并进行反向传播更新参数
        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        # 计算平均损失
        running_loss += loss.item()
        if i % 500 == 499: # 每500次输出一次损失
            with torch.no_grad():
                test_outputs = net(test_image)# 预测值 [batch_size, 10]
                predict_lable = torch.max(test_outputs, 1)[1]
                accuracy = (predict_lable == test_label).sum().item() / test_label.size(0) # 计算准确率
                print('[%d, %5d] loss: %.3f, accuracy: %.3f' % (epoch + 1, i + 1, running_loss / 500, accuracy)) # 输出损失和准确率
                running_loss = 0.0 # 每次损失置0

print('Finished Training')
save_path = './model/letnet.pth'
# 判断是文件夹是否存在
if not os.path.exists('./model'):
    os.mkdir('./model')
torch.save(net.state_dict(), save_path)

