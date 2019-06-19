import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tqdm import tqdm
import operator


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


class Printer(object):

    def __init__(self, ordered_keys):
        self.ordered_keys = ordered_keys
        self.sizes = [max(10, len(key)) for key in self.ordered_keys]

    def print_header(self):
        header = ""
        for key, size in zip(self.ordered_keys, self.sizes):
            header = header + ('{:<%d}' % size).format(key) + "  "
        print(header)

    def print_line(self, measures):
        line = ""
        for key, size in zip(self.ordered_keys, self.sizes):
            line = line + ('{:<%dg}' % size).format(measures[key]) + "  "
        print(line)


class BestModelSaver(object):

    def __init__(self, result_folder, op='lt'):
        """
        :param result_folder (str) path to result folder
        :param op: (str) gt for accuracy, etc, or lt for loss etc.
        """
        self._saver = tf.train.Saver(max_to_keep=None, save_relative_paths=True)
        self._save_path = os.path.join(result_folder, 'best_model.ckpt')
        self._operator = getattr(operator, op)
        self._best_value = float('inf') if op == 'lt' else float('-inf')

    def __call__(self, measure, session):
        if self._operator(measure, self._best_value):
            self._best_value = measure
            self._saver.save(session, self._save_path)


class Logger(object):

    def __init__(self, result_folder='result', name='log'):
        self._log = []
        self._save_path = os.path.join(result_folder, name)

    def add_entry(self, measures):
        self._log.append(measures)

    def write_in_file(self):
        with open(self._save_path, 'w') as file:
            json.dump(self._log, file, indent=4)


def simple_cnn(input_tensor, kernel_size=5, filter_sizes=(20, 50)):
    filters1, filters2 = filter_sizes

    h = slim.conv2d(input_tensor, num_outputs=filters1, kernel_size=kernel_size, stride=1)
    h = slim.batch_norm(h, decay=0.9)
    h = tf.nn.relu(h)
    h = slim.max_pool2d(h, kernel_size=(2, 2), stride=2, padding='same')
    h = slim.conv2d(h, num_outputs=filters2, kernel_size=kernel_size, stride=1)
    h = slim.batch_norm(h, decay=0.9)
    h = tf.nn.relu(h)
    h = slim.max_pool2d(h, kernel_size=(2, 2), stride=2, padding='same')

    h = slim.flatten(h)
    h = slim.linear(h, 500)
    h = slim.linear(h, 10)

    return h


def calc_accuracy(y_, y_conv):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def train(device_id=0):
    if device_id == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    # parameters
    epochs = 10
    batchsize = 32
    result_folder = 'results'
    logger = Logger(result_folder=result_folder)
    float_lr = 0.01
    printer = Printer(['epoch', 'main/loss', 'main/valid/loss', 'main/accuracy', 'main/valid/accuracy', 'elapsed_time'])

    # data preparation
    train_data, test_data, train_label, test_label = get_data(
        style="tensorflow")

    # perform here data augmentation, etc.
    # if operations are done on numpy arrays, you may need to wrap it with tf.py_func()
    def map_fn(_x, _y):
        return _x, _y

    # start graph declaration
    placeholder_data = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    placeholder_label = tf.placeholder(tf.int32, shape=[None])

    # Create separate Datasets for training and validation
    train_dataset = tf.data.Dataset.from_tensor_slices((placeholder_data, placeholder_label))
    train_dataset = train_dataset.batch(batchsize).map(lambda _x, _y: map_fn(_x, _y))
    val_dataset = tf.data.Dataset.from_tensor_slices((placeholder_data, placeholder_label))
    val_dataset = val_dataset.batch(batchsize)

    # Iterator has to have same output types across all Datasets to be used
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    data_x, data_y = iterator.get_next()
    data_y = tf.cast(data_y, tf.int32)

    global_step = tf.Variable(0, trainable=False)
    training_flag = tf.Variable(False, trainable=False)
    learning_rate = tf.Variable(0.001, trainable=False)
    y_ = tf.one_hot(data_y, 10)

    with slim.arg_scope([slim.batch_norm],
                        is_training=training_flag):
        y = simple_cnn(data_x)
    loss = tf.losses.softmax_cross_entropy(y_, y)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_step = slim.learning.create_train_op(loss, optimizer, global_step=global_step)
    accuracy = calc_accuracy(y_, y)

    # Initialize with required Datasets
    train_iterator = iterator.make_initializer(train_dataset)
    val_iterator = iterator.make_initializer(val_dataset)

    bestmodel_saver = BestModelSaver(result_folder)
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        training_start_time = time.time()
        printer.print_header()
        for e in range(epochs):

            metrics_to_log = {'main/accuracy': [], 'main/loss': [],
                              'main/valid/accuracy': [], 'main/valid/loss': []}

            # Start train iterator
            sess.run(train_iterator, feed_dict={placeholder_data: train_data, placeholder_label: train_label})
            try:
                with tqdm(total=train_data.shape[0] // batchsize, leave=False) as pbar:
                    while True:
                        _, main_loss, acc, curr_gt = sess.run([train_step, loss, accuracy, global_step],
                                                              feed_dict={learning_rate: float_lr,
                                                                         training_flag: True})
                        metrics_to_log['main/loss'].append(main_loss)
                        metrics_to_log['main/accuracy'].append(acc)
                        pbar.update(1)
            except tf.errors.OutOfRangeError:
                pass

            # Start validation iterator
            sess.run(val_iterator, feed_dict={placeholder_data: test_data, placeholder_label: test_label})
            try:
                while True:
                    main_loss, acc = sess.run([loss, accuracy])
                    metrics_to_log['main/valid/loss'].append(main_loss)
                    metrics_to_log['main/valid/accuracy'].append(acc)
            except tf.errors.OutOfRangeError:
                pass

            # update learning rate with any function
            float_lr = float_lr * 0.99

            # update statistics, display and save log
            measures = {key: float(np.mean(metrics_to_log[key])) for key in metrics_to_log.keys()}
            elapsed_time = time.time() - training_start_time
            measures['iteration'] = int(curr_gt)
            measures['epoch'] = e
            measures['elapsed_time'] = elapsed_time
            bestmodel_saver(measures['main/valid/loss'], sess)
            printer.print_line(measures)
            logger.add_entry(measures)
            logger.write_in_file()


def predict(device_id=0):
    if device_id == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    train_data, test_data, train_label, test_label = get_data("tensorflow")
    ckpt_path = tf.train.latest_checkpoint('checkpoint/')
    print(ckpt_path)
    x_image = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    y_label = tf.placeholder(tf.int32, shape=[None, ])
    y_ = tf.one_hot(y_label, 10)
    with slim.arg_scope([slim.batch_norm],
                        is_training=False):
        y = simple_cnn(x_image)
    accuracy = calc_accuracy(y_, y)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, ckpt_path)
        acc = sess.run(accuracy, feed_dict={x_image: test_data,
                                            y_label: test_label})
        print("test accuracy", acc)


def main(*args, **kwargs):
    train(*args, **kwargs)

    # call just to check if saved checkpoint is working
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
