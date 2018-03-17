import tensorflow as tf
file = tf.train.match_filenames_once("data.tfrecords-*")
filename_queue = tf.train.string_input_producer(file,shuffle=False)
reader = tf.TFRecordReader()
_,serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'i': tf.FixedLenFeature([],tf.int64),
        'j': tf.FixedLenFeature([],tf.int64),

    })
example,label = features['i'],features['j']
batch_size = 3
capacity = 1000 + 3*batch_size
# example_batch,label_batch = tf.train.batch([example,label],batch_size = batch_size,capacity=capacity)
example_batch,label_batch = tf.train.shuffle_batch([example,label],batch_size = batch_size,min_after_dequeue=30,capacity=capacity)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    for i in range(6):
        cur_example_batch,cur_label_batch = sess.run([example_batch,label_batch])
        print(cur_example_batch,cur_label_batch)
    coord.request_stop()
    coord.join(threads)
