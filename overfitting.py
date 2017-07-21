# Daniel Zduniak

"""Przeuczenie
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def train(data, optimizer, beta=.01):
    images = data.train.images[0:1000]
    labels = data.train.labels[0:1000]

    graph = tf.Graph()

    with graph.as_default():
        # Tworzenie modelu sieci neuronowej
        x = tf.placeholder(tf.float32, [None, 784])  # wejście sieci

        num_hidden = 128  # liczba neuronów warstwy ukrytej
        hW = tf.Variable(tf.truncated_normal([784, num_hidden], stddev=0.1))  # wagi warstwy ukrytej
        hb = tf.Variable(tf.constant(0.1, shape=[num_hidden]))  # wartości progowe warstwy ukrytej
        h = tf.nn.relu(tf.add(tf.matmul(x, hW), hb))

        oW = tf.Variable(tf.zeros([num_hidden, 10]))  # wagi warstwy wyjściowej
        ob = tf.Variable(tf.zeros([10]))  # wartości progowe warstwy wyjściowej

        y = tf.matmul(h, oW) + ob  # wyjście

        # Określenie funckji kosztu i wybór optymizatora
        y_ = tf.placeholder(tf.float32, [None, 10])

        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        regularizer = tf.nn.l2_loss(hW) + tf.nn.l2_loss(oW)
        cost = tf.reduce_mean(cost + beta * regularizer)

        # Dokładność sieci neuronowej
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        train_step = optimizer.minimize(cost)

        sess = tf.InteractiveSession()  # uruchomienie sesji Tensorflow
        tf.global_variables_initializer().run()

        steps = []
        test_accuracy = []
        train_accuracy = []
        loss = []

        # Trenowanie sieci
        for i in range(500):
            ls, _ = sess.run([cost, train_step], feed_dict={x: images, y_: labels})

            if i % 1 == 0:
                steps.append(i)
                acc = sess.run(accuracy, feed_dict={x: images, y_: labels})
                print('Training set accuracy at step %s: %s' % (i, acc))
                train_accuracy.append(acc)
                acc = sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels})
                print('Testing set accuracy at step %s: %s' % (i, acc))
                test_accuracy.append(acc)
                loss.append(ls)

        sess.close()

        return {'steps': steps, 'test': test_accuracy, 'train': train_accuracy, 'loss': loss}


def main(_):
    # Wczytwyanie danych
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    lr = FLAGS.learning_rate

    results = train(mnist, tf.train.AdamOptimizer(lr))
    plt.plot(results['steps'], results['train'], 'r')
    plt.plot(results['steps'], results['test'], 'g')

    results2 = train(mnist, tf.train.AdamOptimizer(lr), .1)
    plt.plot(results2['steps'], results2['test'], 'y')

    plt.figure()
    plt.plot(results['steps'], results['loss'])
    plt.plot(results['steps'], results2['loss'])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Katalog z danymi wejściowymi')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Wartość współczynnika uczenia')
    parser.add_argument('--log_dir', type=str, default='/tmp/neural_test',
                        help='Katalog wyjściowy dla podsumowań dla Tensorboard')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
