"""
Script to convert images to tfrecords format
usage: python images_to_tfrecords.py <list of images> -o <output>
"""
import tensorflow as tf
import scipy.ndimage
import numpy as np
import argparse
import os.path
#import cv2

WIDTH = 92
HEIGHT = 56
DEPTH = 3

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

"""
Save the images to tfrecords
"""
def convert(filenames, output):
    images, labels = read_images(filenames)

    print('Writing output to {}'.format(output))
    with tf.python_io.TFRecordWriter(output) as writer:
        for label, image in zip(labels, images):
            raw = image.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(label),
                'image_raw': _bytes_feature(raw)
            }))
            writer.write(example.SerializeToString())

"""
Read images data, treat the subdirectory as label
"""
def read_images(filenames):
    classes = []
    images = np.zeros((len(filenames), HEIGHT, WIDTH, DEPTH), dtype=np.uint8)
    labels = np.zeros((len(filenames)), dtype=np.int32)

    for i, image in enumerate(filenames):
        class_name = os.path.basename(os.path.dirname(image))

        if class_name not in classes:
            classes.append(class_name)

        class_index = classes.index(class_name)
        
        print('Processing {} as class {}'.format(image, class_index))
        images[i] = scipy.ndimage.imread(image, mode='RGB')
        labels[i] = class_index
        #labels[i] = lab_index

    return images, labels

def main():
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('source', nargs='+', help='list of image(s)')
   # parser.add_argument('label', help='label to be used')
    parser.add_argument('-o', '--output', default='result.tfrecords', help='output filename, default to result.tfrecords')

    args = parser.parse_args()

    # Convert!
    convert(args.source, args.output)

if __name__ == '__main__':
    main()
