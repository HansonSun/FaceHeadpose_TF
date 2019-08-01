import tensorflow  as tf
import os
from tensorflow.python.platform import gfile

pbfile=''
for i in os.listdir("ConvertedModels"):
    if "pb" in i:
        pbfile=i
        break

graph = tf.get_default_graph()
graphdef = graph.as_graph_def()

graphdef.ParseFromString(gfile.FastGFile("ConvertedModels/%s"%pbfile).read())
tf.import_graph_def(graphdef, name="")
sess=tf.Session()
print graph
print sess.graph
summary_write = tf.summary.FileWriter("/home/hanson/work/FaceGlasses_TF/tools//OptimizeModel/tfevents/" , graph=graph)
with open("ConvertedModels/pbnode.txt",'w') as f:
    for node in graphdef.node:
        f.write("name: "+str(node.name)+"  op: "+str(node.op)+"\n" )

sess.close()