import tensorflow as tf
from tensorflow.python.platform import gfile
graph = tf.get_default_graph()
graphdef = graph.as_graph_def()
graphdef.ParseFromString(gfile.FastGFile("model-20180723-182259.ckpt-274.pb", "rb").read())
tf.import_graph_def(graphdef, name="")
summary_write = tf.summary.FileWriter("./" , graph)