from util import *

THRESHOLD = 1

def model(image1, image2, args):
    # Convolution layer #1 (56x46x3) => (50x40x15)
    img1_conv1, img2_conv1 = conv_operations(image1, image2, "conv1", 7, 3, 15)
    pool1_img1, pool1_img2 = pool_operations(img1_conv1, img2_conv1, "pool1", [1,2,2,1], [1,2,2,1]) 

    # Convolution layer #2 (25x20x15) => (20x15x45)
    img1_conv2, img2_conv2 = conv_operations(pool1_img1, pool1_img2, "conv2", 6, 15, 45)
    pool2_img1, pool2_img2 = pool_operations(img1_conv2, img2_conv2, "pool2", [1,4,3,1], [1,4,3,1])

    # Convolutional layer #3 (5x5x45) => (1x1x250)
    img1_conv3, img2_conv3 = conv_operations(pool2_img1, pool2_img2, "conv3", 5, 45, 250)

    # Flatten
    flatten_1 = tf.reshape(img1_conv3, [-1, 1*1*250])
    flatten_2 = tf.reshape(img2_conv3, [-1, 1*1*250])

    # FC layer #1 (1*1*250) => (50)
    fc1_img1, fc1_img2 = fc_operations(flatten_1, flatten_2, "fc_1", 1*1*250, 50, activation=tf.nn.relu)
    fc1_img1 = tf.nn.l2_normalize(fc1_img1, dim=0, epsilon=1e-12)
    fc1_img2 = tf.nn.l2_normalize(fc1_img2,dim= 0, epsilon=1e-12)
    
    return fc1_img1, fc1_img2


def conv_operations(image1, image2, given_name, filter_size, in_size, out_size):
    with  tf.variable_scope("convolution") as scope_conv:
        out_1 = conv_layer(image1, filter_size, in_size, out_size, name=given_name)
        scope_conv.reuse_variables()
        out_2 = conv_layer(image2, filter_size, in_size, out_size, name=given_name)
        
        return out_1, out_2

def conv_layer(in_image, fil_size, no_in, no_out, name):
    with tf.variable_scope(name):
        weights = tf.get_variable("weights", [fil_size, fil_size, no_in, no_out], initializer=tf.random_normal_initializer())
        biases = tf.get_variable("biases", [no_out], initializer=tf.constant_initializer(0.0))
        tf.summary.histogram('weights', weights)
        tf.summary.histogram('biases', biases)
        conv = tf.nn.conv2d(in_image, weights, strides=[1,1,1,1], padding='VALID')
        
        return tf.nn.relu(conv + biases)

def pool_operations(image1, image2, given_name, ksize_value, stride_value):
    with  tf.variable_scope(given_name) as scope_pool:
        pool_1 = tf.nn.max_pool(image1, ksize=ksize_value, strides=stride_value, padding='SAME')
        scope_pool.reuse_variables()
        pool_2 = tf.nn.max_pool(image2, ksize=ksize_value, strides=stride_value, padding='SAME')
        
        return pool_1, pool_2


def fc_operations(image1, image2, given_name, in_size, out_size, activation):
    with tf.variable_scope("pooling") as scope_fc:
        fc_out1 = fc_layer(image1, given_name, in_size, out_size, activation=None)
        scope_fc.reuse_variables()
        fc_out2 = fc_layer(image2, given_name, in_size, out_size, activation=None)
        
        return fc_out1, fc_out2


def fc_layer(image, name, in_size, out_size, activation=None):
    with tf.variable_scope(name):
        weights = tf.get_variable("weights", [in_size, out_size], initializer=tf.random_normal_initializer())
        biases = tf.get_variable("biases", [out_size], initializer=tf.random_normal_initializer(0.0))
        y = tf.add(tf.matmul(image, weights), biases)
        tf.summary.histogram('weights', weights)
        tf.summary.histogram('biases', biases)
        
        if activation is not None:
            y = activation(y)

        return y


def compute_accuracy(distance, true_label, batch_size):
    index_values = distance < THRESHOLD
    places = tf.cast(index_values, tf.float32)
    accuracy = tf.reduce_sum(places)* 100/(batch_size)
    return accuracy
    
    
