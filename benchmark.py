# Daniel Zduniak

"""Program testujący optimiztory gradientowe.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import time

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def train(data, optimizer, name, timing=False):
    if not timing:
        print(name + ':')

    graph = tf.Graph()

    with graph.as_default():
        # Tworzenie modelu sieci neuronowej
        x = tf.placeholder(tf.float32, [None, 784])  # wejście sieci
        W = tf.Variable(tf.zeros([784, 10]))  # wagi
        b = tf.Variable(tf.zeros([10]))  # wartości progowe
        y = tf.matmul(x, W) + b  # wyjście

        # Określenie funckji kosztu i wybór optymizatora
        y_ = tf.placeholder(tf.float32, [None, 10])

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        tf.summary.scalar('loss', cross_entropy)

        # Dokładność sieci neuronowej
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        merged = tf.summary.merge_all()
        train_step = optimizer.minimize(cross_entropy)

        sess = tf.InteractiveSession()  # uruchomienie sesji Tensorflow
        writer = tf.summary.FileWriter(FLAGS.log_dir + '/' + name)
        tf.global_variables_initializer().run()

        t0 = time.time()
        # Trenowanie sieci
        for i in range(1000):
            if not timing and i % 10 == 0:
                summary, acc = sess.run([merged, accuracy], feed_dict={x: data.test.images, y_: data.test.labels})
                if not timing:
                    writer.add_summary(summary, i)
                print('Accuracy at step %s: %s' % (i, acc))
            else:
                batch_xs, batch_ys = data.train.next_batch(100)  # pobieranie mini-pakietu danych
                summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys})
                if not timing:
                    writer.add_summary(summary, i)
        t1 = time.time()

        sess.close()

        return t1 - t0


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Wczytwyanie danych
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    lr = FLAGS.learning_rate

    optimizers = {'gradient_descent': tf.train.GradientDescentOptimizer(lr),
                  'momentum': tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9, use_nesterov=False),
                  'nesterov': tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9, use_nesterov=True),
                  'adagrad': tf.train.AdagradOptimizer(lr),
                  'adadelta': tf.train.AdadeltaOptimizer(1.0),
                  'rmsprop': tf.train.RMSPropOptimizer(lr),
                  'adam': tf.train.AdamOptimizer(lr)}

    for name, optimizer in optimizers.items():
        train(mnist, optimizer, name)

    results = {}
    for name, optimizer in optimizers.items():
        total = 0
        for _ in range(10):
            total += train(mnist, optimizer, name, True)
        avg = total / 10
        results[name] = avg

    for name, avg in results.items():
        print(name + ': ' + '%.2f' % avg + ' s')


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
