import tensorflow as tf
import argparse
from util import *
import time

# Network architecture
from siamese_model import *

MARGIN = 0.5
THRESHOLD = 1

if __name__ == '__main__':
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('train', help='tf record filename')
    parser.add_argument('-b', '--batch-size', default=50, type=int, help='batch size')
    parser.add_argument('-e', '--epochs', default=400, type=int, help='num of epochs')
    parser.add_argument('-o', '--output', default='model', help='model output')
    parser.add_argument('-l', '--log', default='logs', help='log directory')
    args = parser.parse_args()

    # Load datasets

    filename_queue = tf.train.string_input_producer([args.train]) # tf.train.string_input_producer function can combine multiple tf records files 
    # into one single queue runner and subseqently it can process each file inside each tf record files and process them
    image, label = read_and_decode(filename_queue)

    batch = tf.train.shuffle_batch([image, label], batch_size=args.batch_size, capacity=100, num_threads=2, min_after_dequeue=40)

    X1 = tf.placeholder(tf.float32, [None, HEIGHT_NEW, WIDTH_NEW, DEPTH], name='X1')
    X2 = tf.placeholder(tf.float32, [None, HEIGHT_NEW, WIDTH_NEW, DEPTH], name='X2')
    y = tf.placeholder(tf.float32, [None, 1], name='labels')

    y_hat1, y_hat2 = model(X1, X2, {'keep_prob': 1})


    distance = tf.reduce_sum(tf.sqrt(tf.pow(tf.subtract(y_hat1, y_hat2), 2)), 1)

    distance_root = tf.sqrt(distance)

    loss = tf.multiply((1- y), distance) + tf.multiply(y, tf.square((tf.maximum(0.0, MARGIN - distance_root))))

    loss_val = tf.reduce_mean(loss)
    # Train
    train = tf.train.AdamOptimizer(1e-4).minimize(loss_val)

    # calculate true positives
    total_positives = args.batch_size - tf.reduce_sum(y)
    predicted = distance >= THRESHOLD
    actual = tf.cast(y, tf.bool)

    true_positives = args.batch_size - tf.reduce_sum(tf.cast(tf.logical_or(predicted, actual), tf.float32))

    true_positive_prop = true_positives/total_positives

    # calculate false positives
    false_positives = (args.batch_size - tf.reduce_sum(tf.cast(predicted, tf.float32))) - true_positives
    false_positives_prop = false_positives/(tf.reduce_sum(y))

    #tf.summary.scalar('loss', loss)
    #tf.summary.scalar('accu', accuracy)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(args.log, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        roc_file = open("roc_file.txt", "a")

        for i in range(args.epochs):
            images, labels = sess.run(batch)
            images1, images2, labels1 = split_images(images, args.batch_size, labels)
            sess.run(train, feed_dict={X1: images1, X2: images2, y: labels1})
            roc_file.write(str(false_positives_prop))
            roc_file.write(" ")
            roc_file.write(str(true_positives_prop))
            roc_file.write("\n")
            #s = sess.run(merged_summary, feed_dict={X1: images1, X2: images2, y: labels1})
            #writer.add_summary(s, i)
            
            # Print training accuracy every 100 epochs
            if (i+1) % 8 == 0:
                #print(sess.run(loss, feed_dict={X1: images1, X2:images2, y:labels1}))
                print('loss val {}: {:.2f}'.format(i+1, sess.run(loss_val, feed_dict={X1: images1, X2:images2, y: labels1})))
                #print('distance {}: {:.2f}'.format(i+1, sess.run(distance, feed_dict={X1: images1, X2:images2, y: labels1})))
                
            if (i+1) % 1000 == 0:
                params = saver.save(sess, '{}_{}.ckpt'.format(args.output, i+1))
                print('Model saved: {}'.format(params))
                
        coord.request_stop()
        coord.join(threads)
        
        
