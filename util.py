import tensorflow as tf
import argparse
import scipy
import numpy as np 

WIDTH_OLD = 92
HEIGHT_OLD = 56

WIDTH_NEW = 46
HEIGHT_NEW = 56

DEPTH = 3

def read_and_decode(filename_queue):
    with tf.name_scope('read_and_decode'):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })

        label = tf.cast(features['label'], tf.int32)
        label = tf.reshape(label,[1])
            
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        
        # Convert back to image shape
        image = tf.reshape(image, [HEIGHT_OLD, WIDTH_OLD, DEPTH])
        tf.summary.image('real', tf.reshape(image, [1, HEIGHT_OLD, WIDTH_OLD, DEPTH]))
        
        #image = tf.image.resize_images(image, [100, 200])
        image = alter_image(image)
        
        image = tf.cast(image, tf.float32)
        image = tf.image.per_image_standardization(image)
        
        return image, label


def split_images(images, batch_size, labels):
    image1 = images[0:batch_size, 0:56, 0:46, :]
    image2 = images[0:batch_size, 0:56, 46:92, :]
    
    return image1, image2, labels


def alter_image(image_input):
    altered_image = tf.image.resize_images(image_input, [66, 102])
    altered_image = tf.random_crop(altered_image, [56, 92, 3])

        # Randomly flip image
    altered_image = tf.image.random_flip_left_right(altered_image)
        # Randomly adjust brightness
    altered_image = tf.image.random_brightness(altered_image, 50)
    
    return altered_image
