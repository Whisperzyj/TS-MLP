# -*- coding:utf-8 -*-
# author: zyj time: 2021/7/10

import numpy as np

N = 12
d_lamda = 0.5
L = 1000
K = 2
e_ratio = np.arange((N), dtype='complex')

train_snr = np.arange(0, 31, 5)
angle_range = [-60, 60]
delta_angle_range = [1, 10]
train_angle_num = 5000
data_num_per_snr = train_angle_num*((delta_angle_range[1]-delta_angle_range[0])*100)
data_num = data_num_per_snr*len(train_snr)

epochs = 300
batch_size = 1000
learning_rate = 0.0001
split = [0.9, 0.1]
loss_weight = [1, 0.2, 10]

test_num = 1000