###########################################################################################################
##
# main.py
##
# @author Matthew Cline
# @version 20180106
##
# Description: Data examination and deep neural network to analyze radar images and identify icebergs
##
# Data: Data comes for this Kaggle Competition https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data
##
###########################################################################################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import sys
import os
import datetime
import pickle

import tensorflow as tf



####### HYPER PARAMS ########
n_classes = 2
keepRate = 1.0
batchSize = 10

####### TENSORFLOW PLACEHOLDERS #######
x1 = tf.placeholder('float', [None, 5625])
x2 = tf.placeholder('float', [None, 5625])
y = tf.placeholder('float')

weights = {'W_conv1': tf.Variable(tf.truncated_normal([3, 3, 1, 75], stddev=0.1)),
               'W_conv2': tf.Variable(tf.truncated_normal([3, 3, 75, 150], stddev=0.1)),
               'W_conv3': tf.Variable(tf.truncated_normal([3, 3, 150, 75], stddev=0.1)),
               'W_conv4': tf.Variable(tf.truncated_normal([3, 3, 1, 75], stddev=0.1)),
               'W_conv5': tf.Variable(tf.truncated_normal([3, 3, 75, 150], stddev=0.1)),
               'W_conv6': tf.Variable(tf.truncated_normal([3, 3, 150, 75], stddev=0.1)),
               'W_conv1_combo': tf.Variable(tf.truncated_normal([3, 3, 75, 150], stddev=0.1)),
               'W_conv2_combo': tf.Variable(tf.truncated_normal([3, 3, 150, 75], stddev=0.1)),
               'W_fc': tf.Variable(tf.truncated_normal([75 * 75 * 75 * 2, 100], stddev=0.1)),
               'W_out': tf.Variable(tf.truncated_normal([100, n_classes], stddev=0.1))}

biases = {'b_conv1': tf.Variable(tf.zeros([75])),
              'b_conv2': tf.Variable(tf.zeros([150])),
              'b_conv3': tf.Variable(tf.zeros([75])),
              'b_conv4': tf.Variable(tf.zeros([75])),
              'b_conv5': tf.Variable(tf.zeros([150])),
              'b_conv6': tf.Variable(tf.zeros([75])),
              'b_conv1_combo': tf.Variable(tf.zeros([150])),
              'b_conv2_combo': tf.Variable(tf.zeros([75])),
              'b_fc': tf.Variable(tf.zeros([100])),
              'b_out': tf.Variable(tf.zeros([n_classes]))}

######## PLACE TO SAVE THE MODEL AFTER TRAINING ########
modelFn = os.path.normpath('models/tensorflow/iceberg_detector_dual_band_concat.ckpt')
if not os.path.exists(os.path.normpath('models/tensorflow')):
    os.makedirs('models/tensorflow')

####### SET UP LOGGING DIRECTORY FOR TENSORBOARD #######
now = datetime.datetime.now()
runName = str(now.year) + str(now.month) + str(now.day) + "-" +str(now.hour) + str(now.minute)
logFn = os.path.normpath('models/tensorflow/logs/iceberg_detector_dual_band_concat/' + runName)
if not os.path.exists(os.path.normpath('models/tensorflow/logs/iceberg_detector_dual_band_concat')):
    os.makedirs('models/tensorflow/logs/iceberg_detector_dual_band_concat')

####### SET UP OUTPUT DIRECTORY #######
now = datetime.datetime.now()
name = str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute) + ".csv"
outputFn = os.path.normpath('output/dual_band_concat/' + name)
if not os.path.exists(os.path.normpath('output/dual_band_concat')):
    os.makedirs('output/dual_band_concat')

######## UTILITY FUNCTIONS ########
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def convolutional_neural_network(x1, x2, weights, biases):

    x1 = tf.reshape(x1, shape=[-1, 75, 75, 1])
    x2 = tf.reshape(x2, shape=[-1, 75, 75, 1])

    with tf.name_scope('conv1'):
        conv1 = conv2d(x1, weights['W_conv1'] + biases['b_conv1'])
        conv1 = tf.nn.leaky_relu(conv1)
        tf.summary.histogram('activations', conv1)
        conv1 = tf.nn.dropout(conv1, keepRate)

    with tf.name_scope('conv2'):
        conv2 = conv2d(conv1, weights['W_conv2'] + biases['b_conv2'])
        conv2 = tf.nn.leaky_relu(conv2)
        tf.summary.histogram('activations', conv2)
        conv2 = tf.nn.dropout(conv2, keepRate)

    with tf.name_scope('conv3'):
        conv3 = conv2d(conv2, weights['W_conv3'] + biases['b_conv3'])
        conv3 = tf.nn.leaky_relu(conv3)
        tf.summary.histogram('activations', conv3)
        conv3 = tf.nn.dropout(conv3, keepRate)

    with tf.name_scope('conv4'):
        conv4 = conv2d(x2, weights['W_conv4'] + biases['b_conv4'])
        conv4 = tf.nn.leaky_relu(conv4)
        tf.summary.histogram('activations', conv4)
        conv4 = tf.nn.dropout(conv4, keepRate)

    with tf.name_scope('conv4'):
        conv5 = conv2d(conv4, weights['W_conv5'] + biases['b_conv5'])
        conv5 = tf.nn.leaky_relu(conv5)
        tf.summary.histogram('activations', conv5)
        conv5 = tf.nn.dropout(conv5, keepRate)

    with tf.name_scope('conv6'):
        conv6 = conv2d(conv5, weights['W_conv6'] + biases['b_conv6'])
        conv6 = tf.nn.leaky_relu(conv6)
        tf.summary.histogram('activations', conv6)
        conv6 = tf.nn.dropout(conv6, keepRate)

    with tf.name_scope('concatenate'):
        combo = tf.concat([conv3, conv6], axis=0)
        tf.summary.histogram('feature_concatenation', combo)

    with tf.name_scope('conv1_combo'):
        conv1_combo = conv2d(combo, weights['W_conv1_combo'] + biases['b_conv1_combo'])
        conv1_combo = tf.nn.leaky_relu(conv1_combo)
        tf.summary.histogram('activations', conv1_combo)
        conv1_combo = tf.nn.dropout(conv1_combo, keepRate)

    with tf.name_scope('conv2_combo'):
        conv2_combo = conv2d(conv1_combo, weights['W_conv2_combo'] + biases['b_conv2_combo'])
        conv2_combo = tf.nn.leaky_relu(conv2_combo)
        tf.summary.histogram('activations', conv2_combo)
        conv2_combo = tf.nn.dropout(conv2_combo, keepRate)

    with tf.name_scope('fully_connected'):
        fc = tf.reshape(conv2_combo, [-1, 75 * 75 * 75 * 2])
        fc = tf.nn.leaky_relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
        tf.summary.histogram('activations', fc)
        fc = tf.nn.dropout(fc, keepRate)

    with tf.name_scope('output'):
        output = tf.matmul(fc, weights['W_out']) + biases['b_out']

    output = tf.nn.softmax(output)
    return output


def train_network(x1, x2):
    predictions = convolutional_neural_network(x1, x2, weights, biases)

    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        print("\n\n")

    
        print("Loading the model from storage...\n\n")
        saver.restore(sess, modelFn)
        tempEval = []
        for j in range(int(len(compBand1) / batchSize)+1):
            batch_x1 = compBand1[j*batchSize:(j+1)*batchSize]
            batch_x2 = compBand2[j*batchSize:(j+1)*batchSize]
            predOut = predictions.eval(feed_dict={x1: batch_x1, x2: batch_x2})
            if j == 0:
                tempEval = predOut
            else:
                tempEval = np.concatenate((tempEval, predOut))
        tempEval = tempEval[:,1]
        d = {'id': compId, 'is_iceberg': tempEval}
        df = pd.DataFrame(data=d)
        pd.set_option('precision', 3)
        pd.set_option('display.float_format', lambda x: '%.3f' % x)
        print(df)
        print("\n\nPrinting the results to csv file...\n\n")
        df.to_csv(outputFn, index=False)

        plt.figure()
        plt.hist(df['is_iceberg'])
        plt.show()

####### LOAD THE PREPARED DATA FROM THE PICKLE FILES #######
try:
    compBand1 = pickle.load(open("data/pickle/compBand1.p", "rb"))
    compBand2 = pickle.load(open("data/pickle/compBand2.p", "rb"))
    compId = pickle.load(open("data/pickle/compId.p", "rb"))
except:
    print("Problem loading the data from the pickle files... exiting application")
    exit(1)

###### RUN THE SESSION ########
print("\n\nStarting the TensorFlow session...\n\n")
train_network(x1, x2)
