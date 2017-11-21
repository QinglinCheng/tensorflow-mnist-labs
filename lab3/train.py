# !/bash/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from datasets import mnist
from graph import x, y_, accuracy, train_step, loss, \
    summary_w1, summary_b1, summary_w2, summary_b2, summary_learning_rate, \
    writer, summary_accuracy, TRAINING_STEPS, BATCH_SIZE

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
        if i % 100 == 0:
            validate_acc = sess.run(summary_accuracy,
                                    feed_dict=validate_feed)
            writer.add_summary(validate_acc, i)
        writer.add_summary(sess.run(summary_w1), i)
        writer.add_summary(sess.run(summary_b1), i)
        writer.add_summary(sess.run(summary_w2), i)
        writer.add_summary(sess.run(summary_b2), i)
        writer.add_summary(sess.run(summary_learning_rate), i)
        xs, ys = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_step,
                 feed_dict={x: xs, y_: ys})

    test_acc = sess.run(accuracy,
                        feed_dict=test_feed)
    print("After %d traing step(s), test accuracy using average "
          "model is %g" % (TRAINING_STEPS, test_acc))
