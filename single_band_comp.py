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
x = tf.placeholder('float', [None, 5625])
y = tf.placeholder('float')
weights = {'W_conv1': tf.Variable(tf.truncated_normal([3, 3, 1, 75], stddev=0.1)),
               'W_conv2': tf.Variable(tf.truncated_normal([4, 4, 75, 100], stddev=0.1)),
               'W_conv3': tf.Variable(tf.truncated_normal([4, 4, 100, 150], stddev=0.1)),
               'W_conv4': tf.Variable(tf.truncated_normal([4, 4, 150, 100], stddev=0.1)),
               'W_conv5': tf.Variable(tf.truncated_normal([4, 4, 100, 75], stddev=0.1)),
               'W_conv6': tf.Variable(tf.truncated_normal([3, 3, 75, 50], stddev=0.1)),
               'W_conv7': tf.Variable(tf.truncated_normal([3, 3, 50, 25], stddev=0.1)),
               'W_conv8': tf.Variable(tf.truncated_normal([3, 3, 25, 10], stddev=0.1)),
               'W_fc': tf.Variable(tf.truncated_normal([75 * 75 * 10, 1024], stddev=0.1)),
               'W_out': tf.Variable(tf.truncated_normal([1024, n_classes], stddev=0.1))}

biases = {'b_conv1': tf.Variable(tf.zeros([75])),
              'b_conv2': tf.Variable(tf.zeros([100])),
              'b_conv3': tf.Variable(tf.zeros([150])),
              'b_conv4': tf.Variable(tf.zeros([100])),
              'b_conv5': tf.Variable(tf.zeros([75])),
              'b_conv6': tf.Variable(tf.zeros([50])),
              'b_conv7': tf.Variable(tf.zeros([25])),
              'b_conv8': tf.Variable(tf.zeros([10])),
              'b_fc': tf.Variable(tf.zeros([1024])),
              'b_out': tf.Variable(tf.zeros([n_classes]))}

######## PLACE TO SAVE THE MODEL AFTER TRAINING ########
modelFn = os.path.normpath('models/tensorflow/test_single_band_network.ckpt')
if not os.path.exists(os.path.normpath('models/tensorflow')):
    os.makedirs('models/tensorflow')

####### SET UP LOGGING DIRECTORY FOR TENSORBOARD #######
logFn = os.path.normpath('models/tensorflow/logs/test_single_band_network.log')
if not os.path.exists(os.path.normpath('models/tensorflow/logs')):
    os.makedirs('models/tensorflow/logs')

####### SET UP OUTPUT DIRECTORY #######
now = datetime.datetime.now()
name = str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute) + ".csv"
outputFn = os.path.normpath('output/single_band/' + name)
if not os.path.exists(os.path.normpath('output/single_band')):
    os.makedirs('output/single_band')

######## UTILITY FUNCTIONS ########

def conv2d(x, W, b):
    temp = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b
    temp = tf.nn.leaky_relu(temp)
    temp = tf.nn.dropout(temp, keepRate)
    return temp

def convolutional_neural_network(x, weights, biases):

    x = tf.reshape(x, shape=[-1, 75, 75, 1])

    conv1 = conv2d(x, weights['W_conv1'], biases['b_conv1'])
    conv2 = conv2d(conv1, weights['W_conv2'], biases['b_conv2'])
    conv3 = conv2d(conv2, weights['W_conv3'], biases['b_conv3'])
    conv4 = conv2d(conv3, weights['W_conv4'], biases['b_conv4'])
    conv5 = conv2d(conv4, weights['W_conv5'], biases['b_conv5'])
    conv6 = conv2d(conv5, weights['W_conv6'], biases['b_conv6'])
    conv7 = conv2d(conv6, weights['W_conv7'], biases['b_conv7'])
    conv8 = conv2d(conv7, weights['W_conv8'], biases['b_conv8'])


    fc = tf.reshape(conv8, [-1, 75 * 75 * 10])
    fc = tf.nn.leaky_relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keepRate)

    output = tf.matmul(fc, weights['W_out']) + biases['b_out']

    output = tf.nn.softmax(output)

    return output


def train_network(x):
    predictions = convolutional_neural_network(x, weights, biases)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print("\n\n")

        print("Loading the model from storage...\n\n")
        saver.restore(sess, modelFn)
        tempEval = []
        for j in range(int(len(compBand1) / batchSize)+1):
            batch_x = compBand1[j*batchSize:(j+1)*batchSize]
            predOut = predictions.eval(feed_dict={x: batch_x})
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
print("Reading in the test data from pickle files...\n\n")
try:
    compId = pickle.load(open("data/pickle/compId.p", "rb"))
    compBand1 = pickle.load(open("data/pickle/compBand1.p", "rb"))
    compBand2 = pickle.load(open("data/pickle/compBand2.p", "rb"))
except:
    print("Problem loading the data from the pickle files... exiting application")
    exit(1)


###### RUN THE SESSION ########
print("\n\nStarting the TensorFlow session...\n\n")
train_network(x)
