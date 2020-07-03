import os
import tensorflow as tf
import numpy as np
from dataSets import DataSet


class Models(object):
    def __init__(self, modelName, inputs):
        self.inputs = inputs
        self.model_name = modelName
        if self.model_name == 'mnist_2nn':
            self.mnist_2nn_construct(inputs)
        elif self.model_name == 'mnist_cnn':
            self.mnist_cnn_construct(inputs)
        elif self.model_name == 'cifar10_cnn':
            self.cifar10_cnn_construct(inputs)


    def mnist_2nn_construct(self, inputs):
        self.fc1 = self.full_connect(inputs, 784, 200, 'h1')
        self.fc2 = self.full_connect(self.fc1, 200, 200, 'h2')
        self.outputs = self.full_connect(self.fc2, 200, 10, 'last_layer', relu=False)

    def mnist_cnn_construct(self, inputs):
        self.trans_inputs  = tf.reshape(inputs, [-1, 28, 28, 1])
        self.cov1 = self.convolve(self.trans_inputs, 1, 5, 1, 1, 32, 'cov1', True, 'SAME')
        self.pool1 = self.max_pool_nxn(self.cov1, 2, 2, 'pool1')
        self.cov2 = self.convolve(self.pool1, 32, 5, 1, 1, 64, 'cov2', True, 'SAME')
        self.pool2 = self.max_pool_nxn(self.cov2, 2, 2, 'pool2')
        with tf.variable_scope('transform') as scope:
            self.trans_pool2 = tf.reshape(self.pool2, [-1, 7 * 7 * 64])
        self.fc1 = self.full_connect(self.trans_pool2, 7 * 7 * 64, 512, 'fc1')
        self.outputs = self.full_connect(self.fc1, 512, 10, 'last_layer', relu=False)

    def cifar10_cnn_construct(self, inputs):
        self.cov1 = self.convolve(inputs, 3, 5, 1, 1, 64, 'cov1', True, 'SAME')
        self.pool1 = self.max_pool_nxn(self.cov1, 3, 2, 'pool1')
        self.cov2 = self.convolve(self.pool1, 64, 5, 1, 1, 64, 'cov2', True, 'SAME')
        self.pool2 = self.max_pool_nxn(self.cov2, 3, 2, 'pool2')
        with tf.variable_scope('transform') as scope:
            self.trans_pool2 = tf.reshape(self.pool2, [-1, 6 * 6 * 64])
        self.fc1 = self.full_connect(self.trans_pool2, 6 * 6 * 64, 384, 'fc1')
        self.fc2 = self.full_connect(self.fc1, 384, 192, 'fc2')
        self.outputs = self.full_connect(self.fc2, 192, 10, 'last_layer', relu=False)


    def full_connect(self, inputs, num_in, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', shape=[num_in, num_out], dtype=tf.float32, trainable=True)
            biases = tf.get_variable('biases', shape=[num_out], dtype=tf.float32, trainable=True)
            ws_plus_bs = tf.nn.xw_plus_b(inputs, weights, biases)

            if relu == True:
                outputs = tf.nn.relu(ws_plus_bs)
                return outputs
            else:
                return ws_plus_bs


    def convolve(self, inputs, inputs_channels, kernel_size, stride_y, stride_x, num_features, name, relu=True, padding='SAME'):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', shape=[kernel_size, kernel_size, inputs_channels, num_features],
                                      dtype=tf.float32, trainable=True)
            biases = tf.get_variable('baises', shape=[num_features], dtype=tf.float32, trainable=True)
            conv = tf.nn.conv2d(inputs, weights, [1, stride_y, stride_x, 1], padding=padding)
            cov_puls_bs = tf.nn.bias_add(conv, biases)

            if relu == True:
                outputs = tf.nn.relu(cov_puls_bs)
                return outputs
            else:
                return cov_puls_bs


    def max_pool_nxn(self, inputs, ksize, ssize, name):
        with tf.variable_scope(name) as scope:
            return tf.nn.max_pool(inputs, ksize=[1, ksize, ksize, 1], strides=[1, ssize, ssize, 1], padding='SAME')





if __name__=='__main__':

    # GPU preparation
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

    mnist = DataSet('mnist', is_IID=1)

    with tf.variable_scope('inputs') as scope:
        input_images = tf.placeholder(shape=[None, 784], dtype=tf.float32, name='input_images')
        true_label = tf.placeholder(shape=[None, 10], dtype=tf.float32, name='true_label')

    mnist_2nn = Models('mnist_2nn', input_images)
    predict_label = tf.nn.softmax(mnist_2nn.outputs)

    with tf.variable_scope('loss') as scope:
        Cross_entropy = -tf.reduce_mean(true_label*tf.log(predict_label), axis=1)

    with tf.variable_scope('train') as scope:
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = optimizer.minimize(Cross_entropy)

    with tf.variable_scope('validation') as scope:
        correct_prediction = tf.equal(tf.argmax(predict_label, axis=1), tf.argmax(true_label, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    # ---------------------------------------- train --------------------------------------------- #
    with tf.Session(config=tf.ConfigProto(
            log_device_placement=False, \
            allow_soft_placement=True, \
            gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        sess.run(tf.initialize_all_variables())

        for i in range(1000):
            batch_images, batch_labels = mnist.next_batch(100)
            sess.run(train, feed_dict={input_images: batch_images, true_label: batch_labels})
            if i%20 == 0:
                batch_images = mnist.test_data
                batch_labels = mnist.test_label
                print(sess.run(accuracy, feed_dict={input_images: batch_images, true_label: batch_labels}))
