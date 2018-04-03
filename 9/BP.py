import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
SUMMARY_DIR = "E:/pythonfile/study_tensorflow/tensorboard/data/to/log"
BATCH_SIZE = 100
TRAINING_STEPS = 5001
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME="model.ckpt"

def inference(input,regularizer):
    with tf.name_scope("layer1"):
        with tf.name_scope("weight"):
            weights = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
            variable_summaries(weights, 'layer1' + '/weights')
        if regularizer !=None:
            tf.add_to_collection('losses',regularizer(weights))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0,shape=[LAYER1_NODE]))
            variable_summaries(biases,"layer1"+'/biases')
        with tf.name_scope('plus_b'):
            preactivate = tf.matmul(input,weights) + biases
            tf.summary.histogram('layer1' + '/pre_activations',preactivate)
            relu1 = tf.nn.relu(preactivate, name='activation')
            tf.summary.histogram('layer1' + '/activations', relu1)

    with tf.name_scope("layer2"):
        with tf.name_scope("weight"):
            weights = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
            variable_summaries(weights, 'layer2' + '/weights')
        if regularizer !=None:
            tf.add_to_collection('losses',regularizer(weights))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0,shape=[OUTPUT_NODE]))
            variable_summaries(biases,"layer2"+'/biases')
        with tf.name_scope('plus_b'):
            out = tf.matmul(relu1,weights) + biases
            tf.summary.histogram('layer2' + '/pre_activations',out)
            return out

def variable_summaries(var,name ):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name,var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name,mean)
        stddev  = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name,stddev)

def train(mnist):
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, 10], name="y-input")
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x,[-1,28,28,1])
        tf.summary.image('input',image_shaped_input,BATCH_SIZE)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference(x,regularizer=regularizer)
    global_step = tf.Variable(0, trainable=False)
    with tf.name_scope("Moving_Average"):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.name_scope("loss_function"):
        # 交叉熵
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        # 正则L2表达式
        loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
        tf.summary.scalar('loss', loss)
    with tf.name_scope("train_step"):
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                                   mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
        tf.summary.scalar('learning_rate', learning_rate)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            tf.summary.scalar('accuracy',accuracy)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(SUMMARY_DIR, tf.get_default_graph())
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            if i%1000 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # 运行时记录运行信息的proto
                run_metadata = tf.RunMetadata()
                # 将配置信息和记录运行信息的proto传入运行的过程，从而记录运行时每一个节点的时间、空间开销信息
                summary, validate_acc = sess.run([merged, accuracy],
                                                 feed_dict=validate_feed,
                                                 options=run_options, run_metadata=run_metadata)
                # 将节点在运行时的信息写入日志文件
                writer.add_run_metadata(run_metadata, 'step%03d' % i)
                writer.add_summary(summary, i)
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

                print("After %d training step(s),validation accuracy using average mode is %g" % (i, validate_acc))
            else:
                summary, _, loss_Value, step = sess.run([merged, train_op, loss, global_step],
                                                        feed_dict={x: xs, y_: ys})
                print("After %d training step(s),loss on training batch is %g." % (step, loss_Value))
                writer.add_summary(summary, i)
        print("Test accuracy %g" % sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    writer.close()

def main(argv = None):
    mnist = input_data.read_data_sets("./MNIST_data",one_hot=True)
    train(mnist)
if __name__ == '__main__':
    tf.app.run()

