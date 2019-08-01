import tensorflow  as tf
import os
from tensorflow.python.platform import gfile


graphdef = tf.GraphDef()
with gfile.FastGFile("a.pb","rb") as f:
    print f.read()
    graphdef.ParseFromString(f.read())


tf.import_graph_def(graphdef, name="")
with tf.Session() as sess:
    summary_write = tf.summary.FileWriter("./",graph=sess.graph  )
