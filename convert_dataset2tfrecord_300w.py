import sys
import cv2
import itertools
import os
import tensorflow as tf
import PIL.Image
import numpy as np
import random
import datasets
import utils


def tf_int_feature(value):
    if isinstance(value,(list,tuple) ):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def tf_float_feature(value):
    if isinstance(value,(list,tuple) ):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    else:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def tf_bytes_feature(value):
    if isinstance(value,(list,tuple) ):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))




def create_example(img, binned_pose,cont_labels):
    example = tf.train.Example(features=tf.train.Features(feature={
        'img_raw': tf_bytes_feature(img.tobytes() ),
        'img_width':tf_int_feature(img.width),
        'img_height':tf_int_feature(img.height),
        'binned_pose': tf_int_feature(binned_pose),
        'cont_labels': tf_float_feature(cont_labels),
    }))
    return example



if __name__ == '__main__':

    total_img_cnt = 0
    with tf.python_io.TFRecordWriter("tfrecord_dataset/train.tfrecords") as writer:
        testdataset = datasets.Pose_300W_LP()
        for img, binned_labels, cont_labels, imgpath in testdataset.generate():
            total_img_cnt+=1
            img = img.resize((112,112))
            tf_example = create_example(img,binned_labels.tolist(),cont_labels.tolist())
            writer.write(tf_example.SerializeToString())
    print("total img %d"%total_img_cnt)
