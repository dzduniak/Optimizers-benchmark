# Daniel Zduniak

"""Dobieranie współczynnika uczenia.
"""

import tensorflow as tf
import argparse
import sys

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

    i = 0
    for lr in [0.0001, 0.001, 0.01, 0.1, 1]:
        i = i + 1
        train(mnist, tf.train.AdamOptimizer(lr), 'learning_rate_%d' % i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Katalog z danymi wejściowymi')
    parser.add_argument('--log_dir', type=str, default='/tmp/learning_rate',
                        help='Katalog wyjściowy dla podsumowań dla Tensorboard')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
