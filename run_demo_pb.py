import sys
import os
sys.path.append("/home/hanson/faceapp/lib")
from facedetect import facedetect 
import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import gfile
from facedetect import facedetect
import numpy as np
import utils


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 0, keepdims = True)
    s = x_exp / x_sum    
    return s



def load_pb_model(input_graph,modelfile):
    with input_graph.as_default():
        input_graph_def = input_graph.as_graph_def()
        input_graph_def.ParseFromString( gfile.FastGFile(modelfile).read() )
        tf.import_graph_def(input_graph_def, name='')



face_fd = facedetect("tf-ssh")
graph=tf.get_default_graph()
load_pb_model(graph,"headpose.pb")


input_tensor = graph.get_tensor_by_name("input:0")
yaw_output_tensor = tf.get_default_graph().get_tensor_by_name("vgg_16/yaw_fc8/BiasAdd:0")
pitch_output_tensor = tf.get_default_graph().get_tensor_by_name("vgg_16/pitch_fc8/BiasAdd:0")
roll_output_tensor = tf.get_default_graph().get_tensor_by_name("vgg_16/roll_fc8/BiasAdd:0")

sess=tf.Session()

cam=cv2.VideoCapture(0)
idx_tensor = np.arange(0,67)

while(1):
    _,frame=cam.read()

    facerect,face_img=face_fd.findbiggestface(frame)
    if not face_img is None:

            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img,(96,96))
            face_img=face_img.reshape( (1,96,96,3) )
            face_img = face_img.astype(np.float32)
            face_img = (face_img-127.5)/128.0

            yaw, pitch, roll = sess.run([yaw_output_tensor,pitch_output_tensor,
                            roll_output_tensor],feed_dict={input_tensor:face_img})

            # print(yaw.shape)
            yaw = np.squeeze(yaw)
            pitch = np.squeeze(pitch)
            roll = np.squeeze(roll)
            yaw_predicted = softmax(yaw)
            pitch_predicted = softmax(pitch)
            roll_predicted = softmax(roll)

            yaw_predicted = np.sum(yaw_predicted * idx_tensor) * 3 - 99
            pitch_predicted = np.sum(pitch_predicted * idx_tensor) * 3 - 99
            roll_predicted = np.sum(roll_predicted * idx_tensor) * 3 - 99


            print("yaw %f pitch %f roll %f"%(float(yaw_predicted), float(pitch_predicted), float(roll_predicted)))
            utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = 200, tdy= 200, size = 100)
            cv2.imshow("f",frame)
            cv2.waitKey(1)
