import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf


def get_data(style="tensorflow"):
    mnist = tf.keras.datasets.mnist
    (train_data, train_label), (test_data, test_label) = mnist.load_data()

    if style == "chainer":
        train_data = train_data.reshape(-1, 1, 28, 28)
        test_data = test_data.reshape(-1, 1, 28, 28)
    elif style == "tensorflow":
        train_data = train_data.reshape(-1, 28, 28, 1)
        test_data = test_data.reshape(-1, 28, 28, 1)
    else:
        raise Exception(
            "Invalid style = {}. choice chainer or tensorflow".format(style))
    return train_data, test_data, train_label, test_label


class SimpleCNN(tf.keras.models.Model):

    def __init__(self, kernel_size=5, filter_sizes=(20, 50), data_format=None):
        super(SimpleCNN, self).__init__()
        filters1, filters2 = filter_sizes

        if data_format is None or data_format == 'channels_last':
            self.c_axis = -1
        else:
            self.c_axis = 1

        self.conv2a = tf.keras.layers.Conv2D(filters1, kernel_size=kernel_size, strides=1, padding='same',
                                             data_format=data_format)
        self.bn2a = tf.keras.layers.BatchNormalization(axis=self.c_axis)
        self.mp1 = tf.keras.layers.MaxPool2D((2, 2), data_format=data_format)

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size=kernel_size, padding='same', data_format=data_format)
        self.bn2b = tf.keras.layers.BatchNormalization(axis=self.c_axis)
        self.mp2 = tf.keras.layers.MaxPool2D((2, 2), data_format=data_format)

        self.flat = tf.keras.layers.Flatten()

        self.linear1 = tf.keras.layers.Dense(500)
        self.linear2 = tf.keras.layers.Dense(10)

    def call(self, input_tensor, training=False, mask=None, return_feature=False):
        # h = tf.expand_dims(input_tensor, axis=self.c_axis)
        h = input_tensor
        h = self.conv2a(h)
        h = self.bn2a(h, training=training)
        h = tf.nn.relu(h)
        h = self.mp1(h)
        h = self.conv2b(h)
        h = self.bn2b(h, training=training)
        h = tf.nn.relu(h)
        feature = h
        h = self.mp2(h)

        h = self.flat(h)
        h = self.linear1(h)
        h = self.linear2(h)

        if return_feature:
            return h, feature
        else:
            return h


def calc_accuracy(y_, y_conv):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def train(device_id=0):
    if device_id == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device_name = '/device:CPU:0'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        device_name = '/device:GPU:0'  # -> tensorflow counts available devices from 0, regardless to their device id.

    with tf.device(device_name):
        train_data, test_data, train_label, test_label = get_data(
            style="tensorflow")
        x_image = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        y_label = tf.placeholder(tf.int32, shape=[None, ])
        global_step = tf.Variable(0, trainable=False)

        y_ = tf.one_hot(y_label, 10)
        learning_rate = tf.Variable(0.001, trainable=False)
        model = SimpleCNN()
        y = model(x_image, training=True)
        loss = tf.losses.softmax_cross_entropy(y_, y)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_step = optimizer.minimize(loss, global_step=global_step)
        accuracy = calc_accuracy(y_, y)

    # summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=1)
    epochs = 10
    batchsize = 32
    log_list = []
    last_gt = 0
    float_lr = 0.01

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        # summary_writer = tf.summary.FileWriter("logs", sess.graph)
        sess.run(tf.global_variables_initializer())
        training_start_time = time.time()
        for e in range(epochs):

            metrics_to_log = {'main/accuracy': [], 'main/loss': []}
            b = 0

            while b + batchsize < len(train_label):
                data_batch = train_data[b:b + batchsize]
                label_batch = train_label[b:b + batchsize]
                _, main_loss, acc, curr_gt = sess.run([train_step, loss, accuracy, global_step],
                                                      feed_dict={x_image: data_batch,
                                                                 y_label: label_batch,
                                                                 learning_rate: float_lr})
                metrics_to_log['main/loss'].append(main_loss)
                metrics_to_log['main/accuracy'].append(acc)
                b += batchsize

            float_lr = float_lr / 1.5
            measures = {key: float(np.mean(metrics_to_log[key])) for key in metrics_to_log.keys()}

            elapsed_time = time.time() - training_start_time

            speed = (curr_gt - last_gt) / elapsed_time
            print('Epoch {}, Iteration {}, speed {}, '.format(e, curr_gt, speed)
                  + ', '.join(list(map(lambda x: '{}: {}'.format(x, measures[x]), measures.keys()))))

            measures['iteration'] = int(curr_gt)
            measures['epoch'] = e
            measures['elapsed_time'] = elapsed_time
            log_list.append(measures)
            with open('log', 'w') as file:
                json.dump(log_list, file, indent=4)

        saver.save(sess, 'checkpoint/mnist.ckpt')


def predict(device_id=0):
    if device_id == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device_name = '/device:CPU:0'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        device_name = '/device:GPU:0'  # -> tensorflow counts available devices from 0, regardless to their device id.

    with tf.device(device_name):
        train_data, test_data, train_label, test_label = get_data("tensorflow")
        ckpt_path = tf.train.latest_checkpoint('checkpoint/')
        print(ckpt_path)
        x_image = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        y_label = tf.placeholder(tf.int32, shape=[None, ])
        y_ = tf.one_hot(y_label, 10)
        model = SimpleCNN()
        y = model(x_image)
        accuracy = calc_accuracy(y_, y)

    saver = tf.train.Saver()
    print("testlabel", test_label[0])
    with tf.Session() as sess:
        saver.restore(sess, ckpt_path)
        acc = sess.run(accuracy, feed_dict={x_image: test_data,
                                            y_label: test_label})
        print("test accuracy", acc)


def main(*args, **kwargs):
    train(*args, **kwargs)
    tf.reset_default_graph()
    predict(*args, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST example with Tensorflow')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='Device id used for computation. (-1 for cpu)')
    parsed_args = parser.parse_args()
    param = {
        'device_id': parsed_args.gpu
    }
    main(**param)
