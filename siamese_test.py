import tensorflow as tf
import argparse
from util import *

# Network architecture
from siamese_model import *

MARGIN = 0.05




THRESHOLD = 0.15

if __name__ == '__main__':
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('test', help='the testing dataset') # tf records for the test data
    parser.add_argument('model', help='model to used') # .meta file
    parser.add_argument('-b', '--batch-size', default=48, type=int, help='batch size')
    args = parser.parse_args()

    # Load datasets
    filename_queue = tf.train.string_input_producer([args.test])
    image, label = read_and_decode(filename_queue)
    batch = tf.train.shuffle_batch([image, label], batch_size=args.batch_size, capacity=300, num_threads=2, seed=1, min_after_dequeue=40)

    X1 = tf.placeholder(tf.float32, [None, HEIGHT_NEW, WIDTH_NEW, DEPTH], name='X1')
    X2 = tf.placeholder(tf.float32, [None, HEIGHT_NEW, WIDTH_NEW, DEPTH], name='X2')
    y = tf.placeholder(tf.float32, [None, 1], name='labels')

    y_hat1, y_hat2 = model(X1, X2, {'keep_prob': 1})

    distance = tf.reduce_sum((tf.square(tf.subtract(y_hat1, y_hat2))), 1, keep_dims=True)

    distance_root = tf.sqrt(distance)

    loss = tf.multiply((1- y), distance) + tf.multiply(y, tf.square((tf.maximum(0.0, MARGIN - distance_root)) ) )

      # calculate true positives
    total_positives = args.batch_size - tf.reduce_sum(y)
    predicted = distance_root >= THRESHOLD
    actual = tf.cast(y, tf.bool)

   # inter = tf.reduce_sum(tf.cast(tf.logical_or(predicted, actual), tf.float32))

    true_positives = args.batch_size - tf.reduce_sum(tf.cast(tf.logical_or(predicted, actual), tf.float32))
    
    true_positives_prop = tf.divide(true_positives,total_positives)
    
    # calculate false positives
    false_positives = (args.batch_size - tf.reduce_sum(tf.cast(predicted, tf.float32))) - true_positives
    false_positives_prop = tf.divide(false_positives,(tf.reduce_sum(y)))

    # Accuracy Calculation
    match = tf.equal(tf.cast(predicted, tf.float32), y)
    accuracy = (tf.reduce_sum(tf.cast(match, tf.float32))/args.batch_size)*100

    saver = tf.train.Saver()
    
    roc_file = open("roc_file.txt", "a")
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('/u/ramrao/siamese/'+args.model + '.meta')
        new_saver.restore(sess, '/u/ramrao/siamese/'+ args.model)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        tf.set_random_seed(1)
        images, labels = sess.run(batch)
        images1, images2, labels1 = split_images(images, args.batch_size, labels)
        roc_file.write(str(sess.run(false_positives_prop, feed_dict={X1: images1, X2: images2, y: labels1})))
        roc_file.write(" ")
        #roc_file.write(str(sess.run(tf.reduce_sum(y), feed_dict={X1: images1, X2: images2, y: labels1})))
        #roc_file.write(" ")
        roc_file.write(str(sess.run(true_positives_prop, feed_dict={X1: images1, X2: images2, y: labels1})))
        roc_file.write(" ")
        #1roc_file.write(str(sess.run(total_positives, feed_dict={X1: images1, X2: images2, y: labels1})))
        roc_file.write("\n")
        print(sess.run(actual, feed_dict={X1: images1, X2: images2, y: labels1}))
        print(sess.run(predicted, feed_dict={X1: images1, X2: images2, y: labels1}))
        print(sess.run(tf.reduce_mean(loss), feed_dict={X1:images1, X2:images2, y:labels1}))
            
        coord.request_stop()
        coord.join(threads)
