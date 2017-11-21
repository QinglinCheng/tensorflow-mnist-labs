# !/bash/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from datasets import mnist

# ---------- 参数的配置 ----------
# ---------- 输入层的节点数，图片的像素 ----------
INPUT_NODE = 784
# ---------- 输出层的节点数，图片表示的数字（0-9） ----------
OUTPUT_NODE = 10

# ---------- 隐藏层的节点数 ----------
LAYER1_NODE = 500
# ---------- 每次训练的数据个数 ----------
BATCH_SIZE = 100

# ---------- 基础学习率 ----------
LEARNING_RATE_BASE = 0.8
# ---------- 学习率的衰减率 ----------
LEARNING_RATE_DECAY = 0.99

# ---------- 正则化系数 ----------
REGULARAZTION_RATE = 0.0001
# ---------- 训练的总步数 ----------
TRAINING_STEPS = 5000

# ---------- 定义数据节点 ----------
x = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_NODE], name='x')
y_ = tf.placeholder(dtype=tf.float32, shape=[None, OUTPUT_NODE], name='y_')

# ---------- 定义隐藏层的参数 ----------
w1 = tf.Variable(tf.truncated_normal(shape=[INPUT_NODE, LAYER1_NODE],
                                     stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

w2 = tf.Variable(tf.truncated_normal(shape=[LAYER1_NODE, OUTPUT_NODE],
                                     stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

# ---------- f映射关系 ----------
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)
y = tf.matmul(layer1, w2) + b2

global_step = tf.Variable(0, trainable=False)

# ---------- 交叉熵损失函数 ----------
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=tf.argmax(y_, 1), logits=y)

# ---------- 计算交叉熵的平均值 ----------
cross_entropy_mean = tf.reduce_mean(cross_entropy)

# ---------- L2正则化损失函数 ----------
regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
regularaztion = regularizer(w1) + regularizer(w2)

# ---------- 总损失 ----------
loss = cross_entropy_mean + regularaztion

# ---------- 设置衰减的学习率 ----------
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                           global_step,
                                           mnist.train.num_examples/BATCH_SIZE,
                                           LEARNING_RATE_DECAY,
                                           staircase=True)

# ---------- 使用梯度下降优化器来优化损失函数 ----------
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
    loss, global_step)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
