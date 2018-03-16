import matplotlib.pyplot as plt
import tensorflow as tf
image_raw_Data = tf.gfile.FastGFile("1.jpg",'rb').read()
with tf.Session() as sess:
    img = tf.image.decode_jpeg(image_raw_Data)
    # print(img.eval())
    # img = tf.image.convert_image_dtype(img,dtype=tf.float32)
    encoded_img = tf.image.encode_jpeg(img)
    # with tf.gfile.GFile("2.jpg","wb") as f:
    #     f.write(encoded_img.eval())
    # print(type(img))
    # print(type(img))