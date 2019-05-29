import os

import matplotlib
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

matplotlib.use('Agg')

import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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


def mnist_model(x_image):
    h = slim.conv2d(x_image, 8,
                    kernel_size=[3, 3],
                    padding="SAME",
                    biases_initializer=slim.init_ops.zeros_initializer(),
                    normalizer_fn=slim.batch_norm)
    h = tf.nn.relu(h)
    feature = h
    h = slim.max_pool2d(h, [2, 2])
    h = slim.flatten(h)
    y = slim.fully_connected(h, 10)
    return y, feature


def calc_accuracy(y_, y_conv):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def train():
    train_data, test_data, train_label, test_label = get_data(
        style="tensorflow")
    x_image = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    y_label = tf.placeholder(tf.int32, shape=[None, ])

    y_ = tf.one_hot(y_label, 10)
    y, feature = mnist_model(x_image)
    loss = tf.losses.softmax_cross_entropy(y_, y)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_step = optimizer.minimize(loss)
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


def predict():
    train_data, test_data, train_label, test_label = get_data("tensorflow")
    ckpt_path = tf.train.latest_checkpoint('checkpoint/')
    print(ckpt_path)
    x_image = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    y_label = tf.placeholder(tf.int32, shape=[None, ])
    y_ = tf.one_hot(y_label, 10)
    with slim.arg_scope([slim.batch_norm], is_training=False):
        y, feature = mnist_model(x_image)
        accuracy = calc_accuracy(y_, y)

    saver = tf.train.Saver()
    print("testlabel", test_label[0])
    with tf.Session() as sess:
        saver.restore(sess, ckpt_path)
        acc = sess.run(accuracy, feed_dict={x_image: test_data,
                                            y_label: test_label})
        y, feature = sess.run([y, feature], feed_dict={x_image: [test_data[0]],
                                                       y_label: [test_label[0]]})
        print("test accuracy", acc)
    feature = feature[0]
    if not os.path.exists("featuretf"):
        os.mkdir("featuretf")

    feature = feature.transpose(2, 0, 1)
    for idx, plane in enumerate(feature):
        fig, ax = plt.subplots()
        ax.invert_yaxis()
        print(np.max(plane))
        heatmap = ax.pcolor(plane, cmap='gray')
        savepath = os.path.join("featuretf",
                                "feature_{}.png".format(idx))
        fig.savefig(savepath)


def main():
    train()
    tf.reset_default_graph()
    predict()


if __name__ == '__main__':
    main()
