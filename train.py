# Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Loading and Preparing Data (functions)

data_path = "data"
model = "model/cnn.ckpt"
result = "result/"

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def to_RGB(x):
    R = x[0:1024].reshape(32, 32)
    G = x[1024:2048].reshape(32, 32)
    B = x[2048:3072].reshape(32, 32)
    return np.dstack((R, G, B))

def load_data():
    Tr_X = np.asarray([]).reshape([0, 32, 32, 3])
    Tr_Y = []
    Tt_X = np.asarray([]).reshape([0, 32, 32, 3])
    Tt_Y = []
    
    for i in range(5):
        df = "/".join([data_path, 'data_batch_'+str(i+1)])
        batch = unpickle(df)
        img = batch['data']
        lbl = batch['labels']
        for l in lbl:
            oneh = np.zeros([10])
            oneh[int(l)] = 1
            Tr_Y.append(oneh)
        img = np.asarray([to_RGB(img[i]) for i in range(img.shape[0])])
        Tr_X = np.concatenate((Tr_X, img), axis=0)
    Tr_Y = np.asarray(Tr_Y)
    
    df = "/".join([data_path, 'test_batch'])
    batch = unpickle(df)
    img = batch['data']
    lbl = batch['labels']
    for l in lbl:
        oneh = np.zeros([10])
        oneh[int(l)] = 1
        Tt_Y.append(oneh)
    img = np.asarray([to_RGB(img[i]) for i in range(img.shape[0])])
    Tt_X = np.concatenate((Tt_X, img), axis=0)
    Tt_Y = np.asarray(Tt_Y)
    
    return Tr_X, Tr_Y, Tt_X, Tt_Y

def mini_batch(X, Y, bsize):
    for i in range(0, len(X)-bsize+1, bsize):
        indices = slice(i, i+bsize)
        yield X[indices], Y[indices]

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
##############################################################################

#Load Data
trainX, trainY, testX, testY = load_data()

# Normalize Data
trainX = trainX/255.0
testX = testX/255.0

# Design Model
X = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
keep_prob = tf.placeholder(tf.float32)

#CONV_1
W1 = weight_variable([5, 5, 3, 32])
b1 = bias_variable([32])
h1 = tf.nn.relu6(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')+b1) # 32x32x32

#CONV2
W2 = weight_variable([5, 5, 32, 64])
b2 = bias_variable([64])
in_ = tf.nn.dropout(h1, keep_prob)
h2 = tf.nn.relu6(tf.nn.conv2d(in_, W2, strides = [1, 2, 2, 1], padding='SAME')+b2) # 16x16x64
h2_p = tf.nn.max_pool(h2, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME') # 8x8x64

#CONV3
W3 = weight_variable([5, 5, 64, 128])
b3 = bias_variable([128])
in_ = tf.nn.dropout(h2_p, keep_prob)
h3 = tf.nn.relu6(tf.nn.conv2d(in_, W3, strides = [1, 2, 2, 1], padding='SAME')+b3) # 4x4x128
h3_p = tf.nn.max_pool(h3, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME') # 2x2x128

#FC1
Wf1 = weight_variable([2*2*128, 512])
bf1 = bias_variable([512])
h3_p  = tf.reshape(h3_p, [-1, 2*2*128])
in_ = tf.nn.dropout(h3_p, keep_prob)
f1  = tf.nn.relu6(tf.matmul(in_, Wf1)+bf1)

f1 = tf.nn.dropout(f1, keep_prob)

#FC2
Wf2 = weight_variable([512, 10])
bf2 = bias_variable([10])

y_conv = tf.matmul(f1, Wf2)+bf2

y_ = tf.placeholder(tf.float32, [None, 10])

eta = 1e-4
epochs = 1000
saver = tf.train.Saver()
tas = []
vas = []

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(eta).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.device("/gpu:0"):
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(epochs):
            print("Epoch: ", i+1)
            start = time.time()
            for batch in mini_batch(trainX, trainY, 100):
                batch_xs, batch_ys = batch
                feed_dict = {X:batch_xs, y_:batch_ys, keep_prob:0.6} # dropout = 1-keep
                sess.run(train_step, feed_dict=feed_dict)
            print("Time: ", time.time()- start)
            ta,tb = 0,0
            for batch in mini_batch(trainX, trainY, 100):
                batch_xs, batch_ys = batch
                tb+=1
                ta = accuracy.eval({X:batch_xs, y_:batch_ys, keep_prob:1.0}) + ta
            ta = ta/tb

            va,tb = 0,0
            for batch in mini_batch(testX, testY, 100):
                batch_xs, batch_ys = batch
                tb+=1
                va = accuracy.eval({X:batch_xs, y_:batch_ys, keep_prob:1.0}) + va
            va = va/tb
            tas.append(ta*100)
            vas.append(va*100)
            print("Training Accuracy : ",  ta*100)
            print("Test Accuracy : ", va*100)
        saver_path = saver.save(sess, model)
        print('Model saved in file: %s' % saver_path)
########################################################################################
# save updaet log
lg = open(result + 'log_tr_tst.txt', 'w')
lg.write('TA,TST\n')
for i in range(len(tas)):
    lg.write(str(tas[i])+','+str(vas[i])+ '\n')
lg.close()

# save training and validation accuracy in a file
res = open(result + 'Accuracy.txt', 'w')
res.write('Training Accuracy: '+str(ta*100)+'\n')
res.write('Validation Accuracy: '+str(va*100)+'\n')
res.close()
# save the experiment result
plt.title("Training (blue) and Test (green) Accuracy")
plt.xlim([0, epochs+5])
plt.ylim([0, 100])
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.plot(np.arange(0, epochs), tas, 'b')
plt.plot(np.arange(0, epochs), vas, 'g')

plt.savefig(result+"train_test.png")

plt.figure()
plt.title("Training Accuracy")
plt.xlim([0, epochs+5])
plt.ylim([0, 100])
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.plot(np.arange(0, epochs), tas, 'b')
plt.savefig(result+"train.png")

plt.figure()
plt.title("Test Accuracy")
plt.xlim([0, epochs+5])
plt.ylim([0, 100])
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.plot(np.arange(0, epochs), vas, 'g')
plt.savefig(result+"test.png")
