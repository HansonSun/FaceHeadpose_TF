from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import sys
import tensorflow as tf
import importlib
import itertools
import argparse
from tensorflow.python.framework import graph_util
import numpy as np
import tensorflow.contrib.slim as slim



FLAGS = None
sys.path.append("./ToBeConvertModels/net")
net_name=os.listdir("./ToBeConvertModels/net")[0].split(".")[0]

import vgg 

def main():

    network = importlib.import_module(net_name)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 96, 96, 3], name='input')
            #images_placeholder = tf.placeholder(name='input', shape=[None, 96, 96, 3], dtype=tf.float32)
            # Load the model metagraph and checkpoint
            cpkt_file_path=os.path.join("./ToBeConvertModels/checkpoint/0.449233.ckpt")
			# Build the inference graph

            predict_labels = vgg.inference(input_tensor,phase_train=False)

            saver = tf.train.Saver()
            tf.get_default_session().run(tf.global_variables_initializer())
            tf.get_default_session().run(tf.local_variables_initializer())
            saver.restore(tf.get_default_session(), cpkt_file_path)
            # Retrieve the protobuf graph definition and fix the batch norm nodes
            input_graph_def = sess.graph.as_graph_def()
            # Freeze the graph def
            output_graph_def = freeze_graph_def(sess, input_graph_def, ["vgg_16/yaw_fc8/BiasAdd","vgg_16/pitch_fc8/BiasAdd","vgg_16/roll_fc8/BiasAdd"] )
        # Serialize and dump the output graph to the filesystem
        with tf.gfile.GFile("./ConvertedModels/output.pb", 'wb') as f:
            f.write(output_graph_def.SerializeToString())

def freeze_graph_def(sess, input_graph_def, output_node_names):
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'ArgMax':
            print (" network conain ArgMax ops, which is not support by snpe" )

    # Get the list of important nodes
    whitelist_names = []
    for node in input_graph_def.node:
        if (node.name.startswith('vgg_16') ):
            whitelist_names.append(node.name)

    # Replace all the variables in the graph with constants of the same values
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names,
        variable_names_whitelist=whitelist_names)
    return output_graph_def


if __name__ == '__main__':
    main( )

