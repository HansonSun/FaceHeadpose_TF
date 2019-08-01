from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import numpy as np
import tensorflow as tf
from  input_dataset import *
import importlib
import tensorflow.contrib.slim as slim
import time
from datetime import datetime
import shutil
import vgg
import utils
from input_dataset import TFRecordDataset



def run_training():

    #1.create log and model saved dir according to the datetime
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    models_dir = os.path.join("saved_models", subdir, "models")
    if not os.path.isdir(models_dir):  # Create the model directory if it doesn't exist
        os.makedirs(models_dir)
    logs_dir = os.path.join("saved_models", subdir, "logs")
    if not os.path.isdir(logs_dir):  # Create the log directory if it doesn't exist
        os.makedirs(logs_dir)
    topn_models_dir = os.path.join("saved_models", subdir, "topn")#topn dir used for save top accuracy model
    if not os.path.isdir(topn_models_dir):  # Create the topn model directory if it doesn't exist
        os.makedirs(topn_models_dir)
    topn_file=open(os.path.join(topn_models_dir,"topn_acc.txt"),"a+")
    topn_file.close()


    #2.load dataset and define placeholder
    demo=TFRecordDataset(  )
    train_iterator,train_next_element=demo.generateDataset(tfrecord_path='tfrecord_dataset/train.tfrecords',batch_size=512)


    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
    images_placeholder = tf.placeholder(name='input', shape=[None, 96, 96, 3], dtype=tf.float32)
    binned_pose_placeholder = tf.placeholder(name='binned_pose', shape=[None,3 ], dtype=tf.int64)
    cont_labels_placeholder = tf.placeholder(name='cont_labels', shape=[None,3 ], dtype=tf.float32)

    yaw,pitch,roll = vgg.inference(images_placeholder,phase_train=phase_train_placeholder)

    yaw_logit   = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=yaw,labels=binned_pose_placeholder[:,0])
    pitch_logit = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pitch,labels=binned_pose_placeholder[:,1])
    roll_logit  = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=roll,labels=binned_pose_placeholder[:,2])


    loss_yaw   = tf.reduce_mean(yaw_logit)
    loss_pitch = tf.reduce_mean(pitch_logit)
    loss_roll  = tf.reduce_mean(roll_logit)


    softmax_yaw=tf.nn.softmax(yaw)
    softmax_pitch=tf.nn.softmax(pitch)
    softmax_roll=tf.nn.softmax(roll)

    yaw_predicted   =  tf.math.reduce_sum( (softmax_yaw * tf.linspace(0.0,66.0,67) ), 1 )* 3 - 99
    pitch_predicted =  tf.math.reduce_sum( (softmax_pitch * tf.linspace(0.0,66.0,67) ), 1 )* 3 - 99
    roll_predicted  =  tf.math.reduce_sum( (softmax_roll * tf.linspace(0.0,66.0,67) ), 1 )* 3 - 99



    yaw_mse_loss = tf.losses.mean_squared_error(labels=cont_labels_placeholder[:,0], predictions=yaw_predicted)
    pitch_mse_loss = tf.losses.mean_squared_error(labels=cont_labels_placeholder[:,1], predictions=pitch_predicted)
    roll_mse_loss = tf.losses.mean_squared_error(labels=cont_labels_placeholder[:,2], predictions=roll_predicted)

    alpha = 0.001

    total_loss_softmax=(loss_yaw+loss_pitch+loss_roll)
    total_loss_mse = alpha*(yaw_mse_loss+pitch_mse_loss+roll_mse_loss)
    total_loss = total_loss_softmax+total_loss_mse

    

    yaw_correct_prediction = tf.equal(tf.argmax(yaw,1),binned_pose_placeholder[:,0] )
    pitch_correct_prediction = tf.equal(tf.argmax(pitch,1),binned_pose_placeholder[:,1] )
    roll_correct_prediction = tf.equal(tf.argmax(roll,1),binned_pose_placeholder[:,2] )

    yaw_accuracy = tf.reduce_mean(tf.cast(yaw_correct_prediction, tf.float32))
    pitch_accuracy = tf.reduce_mean(tf.cast(pitch_correct_prediction, tf.float32))
    roll_accuracy = tf.reduce_mean(tf.cast(roll_correct_prediction, tf.float32))

    #adjust learning rate
    global_step = tf.Variable(0, trainable=False)
    #learning_rate = tf.train.exponential_decay(0.001,global_step,100000,0.98,staircase=True)
    learning_rate = tf.train.piecewise_constant(global_step, boundaries=[8000, 16000, 24000, 32000], values=[0.001, 0.0001, 0.0001, 0.00001, 0.000001],name='lr_schedule')



    #optimize loss and update
    #optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    #optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss,global_step=global_step)


    saver=tf.train.Saver(tf.trainable_variables(),max_to_keep=5)

    sess=utils.session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    saver.restore(sess, "/home/hanson/work/FaceHeadpose_TF/saved_models/20190801-135403/models/0.563564.ckpt")

    minimum_loss_value=999.0
    total_loss_value = 0.0
    for epoch in range(1000):
        sess.run(train_iterator.initializer)
        while True:
            use_time=0
            try:
                images_train, binned_pose,cont_labels = sess.run(train_next_element)
                start_time=time.time()
                input_dict={phase_train_placeholder:True,images_placeholder:images_train,binned_pose_placeholder:binned_pose,cont_labels_placeholder:cont_labels}
                
                total_loss_mse_value,total_loss_softmax_value,yaw_acc,pitch_acc,roll_acc,step,lr,train_loss,_ = sess.run([
                                        total_loss_mse,
                                        total_loss_softmax,
                                        yaw_accuracy,
                                        pitch_accuracy,
                                        roll_accuracy,
                                        global_step,
                                        learning_rate,
                                        total_loss,
                                        train_op],
                                        feed_dict=input_dict)

                total_loss_value+=train_loss
                end_time=time.time()
                use_time+=(end_time-start_time)

                # display train result
                if(step%100==0):
                    use_time=0
                    average_loss_value = total_loss_value/100.0
                    total_loss_value=0
                    print ("step:%d lr:%f sloss:%f mloss%f average_loss:%f YAW_ACC:%.2f PITCH_ACC:%.2f ROLL_ACC:%.2f epoch:%d"%(step,
                                                                                                           lr,
                                                                                                           total_loss_softmax_value,
                                                                                                           total_loss_mse_value,
                                                                                                           float(average_loss_value),
                                                                                                           yaw_acc,
                                                                                                           pitch_acc,
                                                                                                           roll_acc, 
                                                                                                           epoch) )
                    if average_loss_value<minimum_loss_value:
                        print("save ckpt")
                        filename_cpkt = os.path.join(models_dir,"%f.ckpt"%average_loss_value)
                        saver.save(sess, filename_cpkt)
                        minimum_loss_value=average_loss_value

            except tf.errors.OutOfRangeError:
                print("End of epoch ")
                break


run_training()
