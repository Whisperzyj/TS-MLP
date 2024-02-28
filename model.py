# -*- coding:utf-8 -*-
# author: zyj time: 2021/7/12

import keras
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

class TS_MLP(keras.Model):
    def __init__(self):
        super(TS_MLP, self).__init__()

        self.dense1_1 = layers.Dense(256, activation=tf.nn.relu)
        self.dense1_2 = layers.Dense(512, activation=tf.nn.relu)
        self.dense1_3 = layers.Dense(512, activation=tf.nn.relu)
        self.dense1_4 = layers.Dense(256, activation=tf.nn.relu)
        self.drop = layers.Dropout(0.5)

        self.out1 = layers.Dense(121)

        self.dense2_1 = layers.Dense(512, activation=tf.nn.relu)
        self.dense2_2 = layers.Dense(1024, activation=tf.nn.relu)
        self.dense2_3 = layers.Dense(512, activation=tf.nn.relu)
        self.dense2_4 = layers.Dense(256, activation=tf.nn.relu)
        self.dense2_5 = layers.Dense(128, activation=tf.nn.relu)

        self.out2 = layers.Dense(100)

        self.dense3_1 = layers.Dense(128, activation=tf.nn.relu)
        self.dense3_2 = layers.Dense(64, activation=tf.nn.relu)
        self.dense3_3 = layers.Dense(16, activation=tf.nn.relu)

        self.out3 = layers.Dense(2, activation=tf.nn.sigmoid)

    def call(self, input1):
        x1 = self.dense1_1(input1)
        x1 = self.dense1_2(x1)
        x1 = self.drop(x1)
        x1 = self.dense1_3(x1)
        x1 = self.drop(x1)
        x1 = self.dense1_4(x1)

        output1 = self.out1(x1)

        input2 = layers.concatenate([output1, input1])

        x2 = self.dense2_1(input2)
        x2 = self.dense2_2(x2)
        x2 = self.dense2_3(x2)
        x2 = self.dense2_4(x2)
        x2 = self.dense2_5(x2)

        output2 = self.out2(x2)

        input3 = layers.concatenate([x1, x2])
        x3 = self.dense3_1(input3)
        x3 = self.dense3_2(x3)
        x3 = self.dense3_3(x3)

        output3 = self.out3(x3)

        return output1, output2, output3