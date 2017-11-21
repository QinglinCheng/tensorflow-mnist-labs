# !/bash/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from datasets import mnist
from graph import TRAINING_STEPS, BATCH_SIZE, x, y_, loss, train_step, accuracy

with tf.Session() as sess:
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

    for i in range(TRAINING_STEPS):
        if i % 1000 == 0:
            validate_acc = sess.run(accuracy,
                                    feed_dict=validate_feed)
            print("After %d training step(s), validation accuracy "
                  "using average model is %g" % (i, validate_acc))

        xs, ys = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_step,
                 feed_dict={x: xs, y_: ys})

    test_acc = sess.run(accuracy,
                        feed_dict=test_feed)
    print("After %d traing step(s), test accuracy using average "
          "model is %g" % (TRAINING_STEPS, test_acc))
