import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import cv2
import sys
import math
import numpy as np
import cv2
import utils 

def miximgprocess( img):
    img = tf.image.resize_images(img, [112,112])
    img = tf.random_crop(img, [96, 96, 3])
    # img=tf.cast(img,tf.uint8)
    #img=tf.image.adjust_brightness(img, tf.random_uniform([], self.conf.brightness_range[0], self.conf.brightness_range[1]))
    img = tf.cast(img, tf.float32)  
    img = tf.subtract(img,127.5)
    img = tf.div(img,128.0)
    return img


class TFRecordDataset(object):
  
    def data_parse_function(self,example_proto):
        features = {'img_raw': tf.FixedLenFeature([], tf.string),
                    'img_width':tf.FixedLenFeature([], tf.int64),
                    'img_height':tf.FixedLenFeature([], tf.int64),
                    'binned_pose': tf.FixedLenFeature([3], tf.int64),
                    'cont_labels': tf.FixedLenFeature([3], tf.float32)
                    }

        features = tf.parse_single_example(example_proto, features)
        img_width=tf.cast(features['img_width'], tf.int64)
        img_height=tf.cast(features['img_height'], tf.int64)
        img = tf.decode_raw(features['img_raw'],tf.uint8)
        img = tf.reshape(img, shape=(img_height,img_width,3))
        img=miximgprocess(img)
        binned_pose = tf.cast(features['binned_pose'], tf.int64)
        cont_labels = tf.cast(features['cont_labels'], tf.float32)
        return img,binned_pose,cont_labels



    def generateDataset(self,tfrecord_path,batch_size):  
        record_dataset = tf.data.TFRecordDataset(tfrecord_path)
        record_dataset = record_dataset.map(self.data_parse_function,num_parallel_calls=4)
        record_dataset = record_dataset.shuffle(buffer_size=10000)
        record_dataset = record_dataset.batch(batch_size )
        iterator = record_dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        return iterator,next_element




def read_tfrecord_test():
    demo=TFRecordDataset( )
    iterator,next_element=demo.generateDataset(tfrecord_path="./tfrecord_dataset/train.tfrecords",
                                               batch_size=64)
    sess = tf.Session()

    for i in range(1000):
        sess.run(iterator.initializer)
        while True:
            try:
                images, binned_labels,cont_labels= sess.run(next_element)
                print(binned_labels[0],cont_labels[0])
                resultimg= images[0]
                resultimg=cv2.cvtColor(resultimg,cv2.COLOR_RGB2BGR)
                resultimg = cv2.resize(resultimg, (400,400))
                #utils.plot_pose_cube(nimg, cont_labels[0], cont_labels[1], cont_labels[2], tdx=None, tdy=None, size=150.)
                resultimg =resultimg*128.0+127.5
                resultimg = resultimg.astype(np.uint8)
                utils.plot_pose_cube(resultimg, cont_labels[0][0], cont_labels[0][1], cont_labels[0][2], tdx=200, tdy=200, size=150.)
                #utils.draw_axis(resultimg, cont_labels[0][0], cont_labels[0][1], cont_labels[0][2], tdx=200, tdy=200, size = 100)
                cv2.imshow('test', resultimg)
                cv2.waitKey(0)
            except tf.errors.OutOfRangeError:
                print("End of dataset")
                break


if __name__ == '__main__':
    read_tfrecord_test()