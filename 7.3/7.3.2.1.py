import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))
if __name__ == '__main__':
    num_shards = 2
    instance_per_shard = 2
    for i in range(num_shards):
        filename = ('data.tfrecords-%.5d-of-%.5d'%(i,num_shards))
        writer = tf.python_io.TFRecordWriter(filename)
        for j in range(instance_per_shard):
            example = tf.train.Example(features=tf.train.Features(feature={
                'i': _int64_feature(i),
                'j': _int64_feature(j)
            }))
            writer.write(example.SerializeToString())
    writer.close()



