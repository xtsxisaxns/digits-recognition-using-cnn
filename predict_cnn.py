from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import mahotas
import cv2

import data_process

# parse command line
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required = True,help = "path to image")
args = vars(ap.parse_args())

# load and preprocess image

image = cv2.imread(args["image"])
# image = cv2.imread("data/test1.png")

gray = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)
gray = cv2.equalizeHist(gray)
blured = cv2.GaussianBlur(gray,(5,5),0)


# # (1)Use edge detection to get contours
edged = cv2.Canny(blured,30,150)
cv2.imshow("edged",edged)
cv2.waitKey(0)
(contours,_) = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# # (2)Use binary theresholding to get contours
# ret,binary= cv2.threshold(blured,100,255,cv2.THRESH_BINARY)
# cv2.imshow("binary",binary)
# cv2.waitKey(0)
# (contours,_) = cv2.findContours(binary.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

print("contours length is {}".format(len(contours)))

contours = sorted([(c,cv2.boundingRect(c)[0]) for c in contours],key = lambda x:x[1])


image_size = 28
num_labels = 10
num_channels = 1 # grayscale

# super paramter
batch_size = 1
patch_size = 5
depth1 = 32
depth2 = 64
num_hidden = 512

# define CNN computation graph

tf_test_data = tf.placeholder(tf.float32,shape=(batch_size, image_size, image_size, num_channels))

# model variables.
layer1_weights = tf.get_variable('w1',shape=[patch_size, patch_size, num_channels, depth1],initializer = tf.contrib.layers.xavier_initializer())
layer1_biases = tf.Variable(tf.zeros(depth1))

layer2_weights = tf.get_variable('w2',shape=[patch_size, patch_size, depth1, depth2],initializer = tf.contrib.layers.xavier_initializer())
# layer2_biases = tf.Variable(tf.zeros(depth2))
layer2_biases = tf.Variable(tf.constant(0.1,shape=[depth2]))

layer3_weights = tf.get_variable('w3',shape=[image_size // 4 * image_size // 4 * depth2 , num_hidden],initializer = tf.contrib.layers.xavier_initializer())
# layer3_biases = tf.Variable(tf.zeros(num_hidden))
layer3_biases = tf.Variable(tf.constant(0.1,shape=[num_hidden]))

layer4_weights = tf.get_variable('w4',shape=[num_hidden, num_labels],initializer=tf.contrib.layers.xavier_initializer())
# layer4_biases = tf.Variable(tf.zeros(num_labels))
layer4_biases = tf.Variable(tf.constant(0.1,shape=[num_labels]))

# define model
def model(data,keep_prob = 1.0):
    # conv + max_pool
    conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
    pool = tf.nn.max_pool(conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    hidden = tf.nn.relu(pool + layer1_biases)

    # conv + max_pool
    conv = tf.nn.conv2d(hidden, layer2_weights, [1,1,1, 1], padding='SAME')
    pool = tf.nn.max_pool(conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    hidden = tf.nn.relu(pool + layer2_biases)

    # conv + max_pool
    # conv = tf.nn.conv2d(hidden, new1_weights, [1,1,1,1], padding='SAME')
    # pool = tf.nn.max_pool(conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    # hidden = tf.nn.relu(pool + new1_biases)

    # dropout
    hidden = tf.nn.dropout(hidden,keep_prob=keep_prob)
    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    # hidden = tf.nn.dropout(hidden, keep_prob=keep_prob)
    return tf.matmul(hidden, layer4_weights) + layer4_biases

# predict .
# logits = model(tf_train_dataset,0.5)

test_prediction = tf.nn.softmax(model(tf_test_data))

# add ops to save and restore all variables
saver = tf.train.Saver()

numList = [0,1,2,3,4,5,6,7,8,9]

# predict number
with tf.Session() as sess:
    #restore variables from disk
    saver.restore(sess,"model/cnn_model.ckpt")
    print("Model restored.")

    for (c,_) in contours:
        (x,y,w,h) = cv2.boundingRect(c)

        if x > 3 and y > 3 and w < 200 and h < 200:
            # (x,y,w,h) = (x-2,y-2,w+4,h+4)
            (x, y, w, h) = (x, y, w, h)
        elif h/w < 1:
            continue

        # judge height and width
        # if w > 7 and h > 20:
        if w > 3 and h > 10:
            # # (1) If use Canny edge detection
            roi = gray[y:y+h,x:x+w]
            thresh = roi.copy()
            T = mahotas.thresholding.otsu(roi)
            thresh[thresh > T] = 255
            thresh = cv2.bitwise_not(thresh)

            # # (2) If use binary theresholding
            # thresh = binary[y:y+h,x:x+w]
            # thresh[thresh > 100] = 255
            # thresh = cv2.bitwise_not(thresh)

            # deskew and centerize
            thresh = data_process.deskew(thresh,28)
            thresh = data_process.center_extent(thresh,(28,28))
            thresh = cv2.resize(thresh,(28,28))

            # reshape to input format of cnn model
            thresh = thresh.reshape(
            		(-1, image_size, image_size, num_channels)).astype(np.float32)

            # predict number
            feed_dict = {tf_test_data : thresh}
            predictions = sess.run([test_prediction], feed_dict=feed_dict)
            numIndex = np.argmax(predictions)
            preDigit = numList[numIndex]

            print("predicted number is {}".format(preDigit))
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
            cv2.putText(image,str(preDigit),(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,0,0),1)


cv2.imshow("image",image)
cv2.imwrite("data/test_result.png",image)
cv2.waitKey(0)
