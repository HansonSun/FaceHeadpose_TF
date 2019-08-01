import tensorflow as tf
converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph("headpose.pb",  
    ['input'], ["vgg_16/yaw_fc8/BiasAdd","vgg_16/pitch_fc8/BiasAdd","vgg_16/roll_fc8/BiasAdd"] )

tflite_model = converter.convert()
open("model.tflite", "wb").write(tflite_model)