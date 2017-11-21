# !/bash/bin/env python
# -*- coding: utf-8 -*-
import os

import tensorflow as tf

from datasets import mnist
from graph import x, y_, TRAINING_STEPS, BATCH_SIZE, train_step, global_step, \
    MODEL_PATH, MODEL_NAME


with tf.Session() as sess:
    # --------- 初始化tf.train.Saver类实例saver用于保存模型 ----------
    saver = tf.train.Saver()

    # --------- 初始化变量 ---------
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    validate_feed = {
        x: mnist.validation.images,
        y_: mnist.validation.labels
    }

    test_feed = {
        x: mnist.test.images,
        y_: mnist.test.labels
    }

    for i in range(TRAINING_STEPS+1):
        if i % 1000 == 0:
            saver.save(sess,
                       os.path.join(MODEL_PATH, MODEL_NAME),
                       global_step=global_step)

        xs, ys = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_step,
                 feed_dict={x: xs, y_: ys})
