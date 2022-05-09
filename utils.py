#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @IDE          :PyCharm
# @Project      :PytorchDemo_LetNet
# @FileName     :utils
# @CreatTime    :2022/5/9 11:33 
# @CreatUser    :DaneSun
import numpy as np
from matplotlib import pyplot as plt


def img_show(img):
    img = img / 2 + 0.5     # 反标准化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()