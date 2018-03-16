import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
def distort_color(image,color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32 /255)
        image = tf.image.random_saturation(image, lower=0.5,upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower = 0.5,upper = 1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
        image = tf.image.random_brightness(image,max_delta=32/255)
        image = tf.image.random_contrast(image, lower = 0.5,upper = 1.5)
        image = tf.image.random_hue(image,max_delta=0.2)
    elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32/255 )
        image = tf.image.random_hue(image, max_delta=0.2)
    return image

def preprocess_for_train(image,height,width,bbox):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image,dtype=tf.float32)
    bbox_begin,bbox_size,_ = tf.image.sample_distorted_bounding_box(tf.shape(image),bounding_boxes=bbox)
    distorted_image = tf.slice(image,bbox_begin,bbox_size)
    image = tf.image.resize_images(image,(height,width),method=np.random.randint(4))
    image = tf.image.random_flip_left_right(image)
    distorted_image = distort_color(image,np.random.randint(2))
    return image
def pre_main(img,bbox = None):
    if bbox is None:
        bbox = tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[1,1,4])
    with tf.gfile.FastGFile(img,'rb') as f:
        image_raw_Data = f.read()
    with tf.Session() as sess:
        image = tf.image.decode_jpeg(image_raw_Data)
        image = preprocess_for_train(image, 299, 299, bbox)

        for i in range(6):
            image = preprocess_for_train(image, 299, 299, bbox)
            plt.imshow(image.eval())
            plt.show()
if __name__ == '__main__':
    pre_main("../path/pic/1.jpg",bbox=None)
    exit()

