import tensorflow as tf
import tensorflow.contrib.slim as slim


def inference(inputs,
            phase_train=True,
            keep_probability=0.5,
            weight_decay=0.0005,
            scope='vgg_16',
            w_init=slim.xavier_initializer_conv2d(uniform=True)):
    end_points={}
    with tf.variable_scope(scope, 'vgg_16', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
            weights_initializer=w_init,
            weights_regularizer=slim.l2_regularizer(weight_decay)):

            net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            net = slim.conv2d(net, 4096,[1,1], padding="VALID", scope='fc6')
            net = slim.dropout(net, keep_probability, is_training=phase_train,scope='dropout6')
            net = slim.conv2d(net, 4096,[1,1], scope='fc7')
            net = slim.dropout(net, keep_probability, is_training=phase_train,scope='dropout7')

            net = slim.flatten(net)
            yaw = slim.fully_connected(net, 67, activation_fn=None,scope='yaw_fc8')
            pitch = slim.fully_connected(net, 67, activation_fn=None,scope='pitch_fc8')
            roll = slim.fully_connected(net, 67, activation_fn=None,scope='roll_fc8')

            # yaw=slim.conv2d(net, 67, [3, 3],activation_fn=None,normalizer_fn=None,scope='yaw_fc8')
            # pitch=slim.conv2d(net, 67, [3, 3],activation_fn=None,normalizer_fn=None,scope='pitch_fc8')
            # roll=slim.conv2d(net, 67, [3, 3],activation_fn=None,normalizer_fn=None,scope='roll_fc8')


            return yaw,pitch,roll
