import argparse
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim


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
        device_name = '/device:CPU:0'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        device_name = '/device:GPU:0'  # -> tensorflow counts available devices from 0, regardless to their device id.

    with tf.device(device_name):
        train_data, test_data, train_label, test_label = get_data(
            style="tensorflow")
        x_image = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        y_label = tf.placeholder(tf.int32, shape=[None, ])

        y_ = tf.one_hot(y_label, 10)
        with slim.arg_scope([slim.batch_norm],
                            is_training=True):
            y = simple_cnn(x_image)
        loss = tf.losses.softmax_cross_entropy(y_, y)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_step = slim.learning.create_train_op(loss, optimizer)
        accuracy = calc_accuracy(y_, y)
    saver = tf.train.Saver(max_to_keep=1)
    epochs = 10
    batchsize = 32
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            b = 0
            while b + batchsize < len(train_label):
                data_batch = train_data[b:b + batchsize]
                label_batch = train_label[b:b + batchsize]
                _, acc = sess.run([train_step, accuracy],
                                  feed_dict={x_image: data_batch,
                                             y_label: label_batch})
                b += batchsize
            print(acc)
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
        with slim.arg_scope([slim.batch_norm],
                            is_training=False):
            y = simple_cnn(x_image)
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
