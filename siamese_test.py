import tensorflow as tf
import argparse
from util import *

# Network architecture
from siamese_model import *

if __name__ == '__main__':
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('test', help='the testing dataset') # tf records for the test data
    #parser.add_argument('model', help='model to used') # .meta file
    args = parser.parse_args()

    # Load datasets
    filename_queue = tf.train.string_input_producer([args.test])
    image, label = read_and_decode(filename_queue)
    batch = tf.train.batch([image, label], batch_size=120)

    X1 = tf.placeholder(tf.float32, [None, HEIGHT_NEW, WIDTH_NEW, DEPTH], name='X1')
    X2 = tf.placeholder(tf.float32, [None, HEIGHT_NEW, WIDTH_NEW, DEPTH], name='X2')
    y = tf.placeholder(tf.float32, [None, 1], name='labels')

    y_hat1, y_hat2 = model(X1, X2, {'keep_prob': 1})

    distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(y_hat1, y_hat2), 2), 1, keep_dims=True))
    distance_val = tf.reduce_mean(distance)

    loss= tf.add( tf.multiply(y, tf.square(tf.maximum(0., 0.2 - distance))), tf.multiply((1- y), tf.pow(distance, 2)))

    accuracy = compute_accuracy(loss, y, 120)

    # correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('/u/ramrao/siamese/model_1000.ckpt.meta')
        new_saver.restore(sess, '/u/ramrao/siamese/model_1000.ckpt')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        images, labels = sess.run(batch)
        images1, images2, labels1 = split_images(images, 120, labels)
        print(sess.run(loss, feed_dict={X1: images1, X2: images2, y: labels1}))
        print('Test Accuracy: {:.2f}'.format(sess.run(accuracy, feed_dict={X1: images1, X2: images2, y: labels1})))

        coord.request_stop()
        coord.join(threads)