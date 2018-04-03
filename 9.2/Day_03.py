import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99

REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 2000
MOVING_AVERAGE_DECAY = 0.99
def get_weight_Variable(shape,regularizer):
    weights = tf.get_variable("weights",shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection("losses",regularizer(weights))
    return weights
def inference(input_tensor,regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_Variable([INPUT_NODE,LAYER1_NODE],regularizer)
        biases = tf.get_variable("biases",[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
    with tf.variable_scope('layer2'):
        weights = get_weight_Variable([LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases = tf.get_variable("biases",[OUTPUT_NODE],initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1,weights)+biases
    return layer2
def train(mnist):
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-input")

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference(x, regularizer)
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

    with tf.name_scope("train_Step"):
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                                   mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
        with tf.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.no_op(name='train')
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            for i in range(TRAINING_STEPS):
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                _, loss_Value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
                print("After %d training step(s),loss on training batch is %g." % (step, loss_Value))


    writer = tf.summary.FileWriter("E:/pythonfile/study_tensorflow/tensorboard/data/to/log",tf.get_default_graph())
    writer.close()
def main(argv = None):
    mnist = input_data.read_data_sets("./MNIST_data",one_hot=True)
    train(mnist)
if __name__ == '__main__':
    tf.app.run()

