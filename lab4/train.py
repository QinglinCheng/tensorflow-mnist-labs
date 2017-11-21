# !/bash/bin/env python
# -*- coding: utf-8 -*-
import argparse
import sys

import tensorflow as tf

from datasets import mnist
from graph import build_graph, TRAINING_STEPS, MODEL_PATH, BATCH_SIZE

FLAGS = tf.app.flags.FLAGS


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # 使用参数服务器和工作节点创建集群
    cluster = tf.train.ClusterSpec({"ps": ps_hosts,
                                    "worker": worker_hosts})

    # 创建并启动服务
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()

    elif FLAGS.job_name == "worker":

        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
            x, y_, train_step, accuracy = build_graph()

        hooks = [tf.train.StopAtStepHook(last_step=TRAINING_STEPS),
                 tf.train.CheckpointSaverHook(checkpoint_dir=MODEL_PATH,
                                              save_secs=600,
                                              saver=tf.train.Saver(sharded=True))]
        is_chief = (FLAGS.task_index == 0)
        i = 1
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }

        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=is_chief,
                                               checkpoint_dir=MODEL_PATH,
                                               hooks=hooks) as mon_sess:
            while not mon_sess.should_stop():
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                mon_sess.run(train_step, feed_dict={x: xs, y_: ys})
                i += 1
                if i % 100 == 0:
                    validate_acc = mon_sess.run(accuracy,
                                                feed_dict=validate_feed)
                    print("After %d training step(s), validation accuracy "
                          "using average model is %g" % (i, validate_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )

    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )

    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
