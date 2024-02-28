# -*- coding:utf-8 -*-
# author: zyj time: 2021/7/12

import os
from parameter import *

import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import io
from dataset import dataset_create
from data_loader import load_data
from label_process import l_preprocess
from model import TS_MLP

from keras.callbacks import ReduceLROnPlateau
from keras import layers, Input, optimizers, callbacks, losses

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
IDS = [1]


tbcallbacks =[
    callbacks.TensorBoard(
        log_dir='my_log_dir',
        histogram_freq=1,
        embeddings_freq=1,
        write_grads=True,
    )
]

data_path = 'data/'
label_path = 'label/'
# dataset_create(data_path, label_path, N, K, angle_range, delta_angle_range)

(data, label) = load_data(data_path, label_path, len(train_snr), N, K, train_snr, data_num)

x = data
y1, y2, y3 = l_preprocess(label)

TS_MLP_net = TS_MLP()

def main():
    TS_MLP_net.build(input_shape=(None, N*N))
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=1, factor=0.7)
    TS_MLP_net.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss=['mse', 'mse', 'binary_crossentropy'], loss_weights=loss_weight, metrics=['acc'])
    history = TS_MLP_net.fit(x, [y1, y2, y3], epochs=epochs, batch_size=batch_size, verbose=1, validation_split=split[1], callbacks=[tbcallbacks, learning_rate_reduction])
    io.savemat('loss.mat', {'total_loss': history.history['loss'], 'integer_loss': history.history['output_1_loss'], 'decimal_loss': history.history['output_2_loss'],
                                'pair_loss': history.history['output_3_loss']})
    TS_MLP_net.save('TS-MLP')
    plt.figure(1)
    plt.plot(np.array(history.history['loss']))
    plt.plot(np.array(history.history['val_loss']))
    plt.xlabel('Epoch')
    plt.ylabel('Train MSE')
    plt.show()

if __name__ == '__main__':
    main()
