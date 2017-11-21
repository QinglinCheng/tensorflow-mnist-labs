# !/bash/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data

# path_to_mnist_data = '/root/data'
path_to_mnist_data = '/root/data'
mnist = input_data.read_data_sets(path_to_mnist_data, one_hot=True)
