# !/bash/bin/env python
# -*- coding: utf-8 -*-
import time

import tensorflow as tf

from datasets import mnist
from graph import x, y_, accuracy, MODEL_PATH

EVAL_INTERVAL = 5

with tf.Session() as sess:
    # --------- 初始化tf.train.Saver类实例saver用于读取模型 ----------
    saver = tf.train.Saver()

    validate_feed = {
        x: mnist.validation.images,
        y_: mnist.validation.labels
    }

    while True:
        ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split(
                '/')[-1].split('-')[-1]

            validate_acc = sess.run(accuracy, feed_dict=validate_feed)
            print("After %s training step(s), validation accuracy "
                  "using average model is %g" % (global_step, validate_acc))

            time.sleep(EVAL_INTERVAL)
        else:
            print('No checkout point file found')
            break
