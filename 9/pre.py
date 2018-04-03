import cv2
import numpy as np
import os
import tensorflow as tf
import time
from TMP import BP
filename = os.listdir("img")
filename.sort(key = lambda x:int(x[:-4]))
for i in filename:
    imgPath = "img/" + i
    img = cv2.imread(imgPath,0)
    input = np.reshape(img,(1,img.shape[0]*img.shape[1]))
    with tf.Graph().as_default() as g:
        input_tensor = tf.cast(input, tf.float32)
        logit = BP.inference(input_tensor, None)
        q = tf.nn.softmax(logit)
        max_index = tf.argmax(q, 1)
        acc_q = q[0][max_index[0]]
        variable_averages = tf.train.ExponentialMovingAverage(BP.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)
        with tf.Session() as sess:
            tf.local_variables_initializer().run()
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            ckpt = tf.train.get_checkpoint_state(BP.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                a, b = sess.run([max_index, acc_q])
                print(a)
            else:
                print("No checkpoint file found")




