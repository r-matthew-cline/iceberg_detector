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
learningRate = 0.001
beta = 0.01
n_classes = 2
keepRate = 0.7
batchSize = 10
numEpochs = 10000
displayStep = 5

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
    # weights = {'W_conv1': tf.Variable(tf.random_normal([3, 3, 1, 75])),
    #            'W_conv2': tf.Variable(tf.random_normal([3, 3, 75, 150])),
    #            'W_conv3': tf.Variable(tf.random_normal([3, 3, 150, 75])),
    #            'W_conv4': tf.Variable(tf.random_normal([3, 3, 1, 75])),
    #            'W_conv5': tf.Variable(tf.random_normal([3, 3, 75, 150])),
    #            'W_conv6': tf.Variable(tf.random_normal([3, 3, 150, 75])),
    #            'W_conv1_combo': tf.Variable(tf.random_normal([3, 3, 75, 150])),
    #            'W_conv2_combo': tf.Variable(tf.random_normal([3, 3, 150, 75])),
    #            'W_fc': tf.Variable(tf.random_normal([75 * 75 * 75 * 2, 100])),
    #            'W_out': tf.Variable(tf.random_normal([100, n_classes]))}

    # biases = {'b_conv1': tf.Variable(tf.random_normal([75])),
    #           'b_conv2': tf.Variable(tf.random_normal([150])),
    #           'b_conv3': tf.Variable(tf.random_normal([75])),
    #           'b_conv4': tf.Variable(tf.random_normal([75])),
    #           'b_conv5': tf.Variable(tf.random_normal([150])),
    #           'b_conv6': tf.Variable(tf.random_normal([75])),
    #           'b_conv1_combo': tf.Variable(tf.random_normal([150])),
    #           'b_conv2_combo': tf.Variable(tf.random_normal([75])),
    #           'b_fc': tf.Variable(tf.random_normal([100])),
    #           'b_out': tf.Variable(tf.random_normal([n_classes]))}

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

    return output


def train_network(x1, x2):
    predictions = convolutional_neural_network(x1, x2, weights, biases)

    with tf.name_scope('Cost'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predictions))
        tf.summary.scalar('Cross Entropy', cost)
        # with tf.name_scope('Regularization'):
        #     regularizer = tf.nn.l2_loss(weights['W_conv1']) + tf.nn.l2_loss(weights['W_conv2']) + tf.nn.l2_loss(weights['W_conv3']) + tf.nn.l2_loss(weights['W_conv4']) + tf.nn.l2_loss(weights['W_conv5']) + tf.nn.l2_loss(weights['W_conv6']) + tf.nn.l2_loss(weights['W_conv1_combo']) + tf.nn.l2_loss(weights['W_conv2_combo']) + tf.nn.l2_loss(weights['W_fc']) + tf.nn.l2_loss(weights['W_out'])
        #     cost = tf.reduce_mean(cost + beta * regularizer)
        #     tf.summary.scalar('Reg_cost', cost)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logFn)
        writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())

        print("\n\n")

        if sys.argv[1] == 'train':
            print("Beginning the training of the model...\n\n")

            for epoch in range(1, numEpochs + 1):
                epochLoss = 0
                epoch_x1 = 0
                epoch_x2 = 0
                epoch_y = 0
                for j in range(int(len(trainBand1) / batchSize)+1):
                    epoch_x1 = trainBand1[j*batchSize:(j+1)*batchSize]
                    epoch_x2 = trainBand2[j*batchSize:(j+1)*batchSize]
                    epoch_y = trainLabels[j*batchSize:(j+1)*batchSize]
                    _, c = sess.run([optimizer, cost], feed_dict={x1: epoch_x1, x2: epoch_x2, y: epoch_y})
                    epochLoss += c
                if epoch % displayStep == 0 or epoch == 1:
                    summary = sess.run(merged, feed_dict={x1: epoch_x1, x2: epoch_x2, y: epoch_y})
                    writer.add_summary(summary, epoch)
                    print("Epoch ", epoch, ' completed out of ', numEpochs, ' with loss: ', epochLoss)
                    save_path = saver.save(sess, modelFn)
                    print("Model saved in file: %s" % save_path)

            print("Training Finished Successfully...")
            save_path = saver.save(sess, modelFn)
            print("Model saved in file: %s" % save_path)

            tempEval = []
            for j in range(int(len(testBand1) / batchSize)+1):
                batch_x1 = testBand1[j*batchSize:(j+1)*batchSize]
                batch_x2 = testBand2[j*batchSize:(j+1)*batchSize]
                predOut = predictions.eval(feed_dict={x1: batch_x1, x2: batch_x2})
                if j == 0:
                    tempEval = predOut
                else:
                    tempEval = np.concatenate((tempEval, predOut))
            predOut = np.argmax(tempEval, axis=1)
            tempLabels = np.argmax(testLabels, axis=1)
            print("Predictions: ", predOut.shape, "\n\n")
            print("Labels: ", tempLabels.shape, "\n\n")
            tn, fp, fn, tp = skm.confusion_matrix(tempLabels, predOut).ravel()
            print("\n\nConfusion Matrix:\n", skm.confusion_matrix(tempLabels, predOut), "\n\n")
            print("True Negative: ", tn)
            print("False Negative: ", fn)
            print("False Positive: ", fp)
            print("True Positive: ", tp)
            print("Accuracy: ", skm.accuracy_score(tempLabels, predOut), "%\n")
            print("Precision: ", skm.precision_score(tempLabels, predOut))
            print("Recall: ", skm.recall_score(tempLabels, predOut))
            print("F1 Score: ", skm.f1_score(tempLabels, predOut))
        ####### Continue Training from a saved model #######
        elif sys.argv[1] == 'continue':
            print("Loading the model from storage to continue training...\n\n")
            saver.restore(sess, modelFn)
            for epoch in range(1, numEpochs + 1):
                epochLoss = 0
                epoch_x1 = 0
                epoch_x2 = 0
                epoch_y = 0
                for j in range(int(len(trainBand1) / batchSize)+1):
                    epoch_x1 = trainBand1[j*batchSize:(j+1)*batchSize]
                    epoch_x2 = trainBand2[j*batchSize:(j+1)*batchSize]
                    epoch_y = trainLabels[j*batchSize:(j+1)*batchSize]
                    _, c = sess.run([optimizer, cost], feed_dict={x1: epoch_x1, x2: epoch_x2, y: epoch_y})
                    epochLoss += c
                if epoch % displayStep == 0 or epoch == 1:
                    summary = sess.run(merged, feed_dict={x1: epoch_x1, x2: epoch_x2, y: epoch_y})
                    writer.add_summary(summary, epoch)
                    print("Epoch ", epoch, ' completed out of ', numEpochs, ' with loss: ', epochLoss)
                    save_path = saver.save(sess, modelFn)
                    print("Model saved in file: %s" % save_path)

            print("Training Finished Successfully...")
            save_path = saver.save(sess, modelFn)
            print("Model saved in file: %s" % save_path)

            tempEval = []
            for j in range(int(len(testBand1) / batchSize)+1):
                batch_x1 = testBand1[j*batchSize:(j+1)*batchSize]
                batch_x2 = testBand2[j*batchSize:(j+1)*batchSize]
                predOut = predictions.eval(feed_dict={x1: batch_x1, x2: batch_x2})
                if j == 0:
                    tempEval = predOut
                else:
                    tempEval = np.concatenate((tempEval, predOut))
            predOut = np.argmax(tempEval, axis=1)
            tempLabels = np.argmax(testLabels, axis=1)
            print("Predictions: ", predOut.shape, "\n\n")
            print("Labels: ", tempLabels.shape, "\n\n")
            tn, fp, fn, tp = skm.confusion_matrix(tempLabels, predOut).ravel()
            print("\n\nConfusion Matrix:\n", skm.confusion_matrix(tempLabels, predOut), "\n\n")
            print("True Negative: ", tn)
            print("False Negative: ", fn)
            print("False Positive: ", fp)
            print("True Positive: ", tp)
            print("Accuracy: ", skm.accuracy_score(tempLabels, predOut), "%\n")
            print("Precision: ", skm.precision_score(tempLabels, predOut))
            print("Recall: ", skm.recall_score(tempLabels, predOut))
            print("F1 Score: ", skm.f1_score(tempLabels, predOut))
        ####### Run only the test data throught the model #######
        else:
            print("Loading the model from storage...\n\n")
            saver.restore(sess, modelFn)
            tempEval = []
            for j in range(int(len(testBand1) / batchSize)+1):
                batch_x1 = testBand1[j*batchSize:(j+1)*batchSize]
                batch_x2 = testBand2[j*batchSize:(j+1)*batchSize]
                predOut = predictions.eval(feed_dict={x1: batch_x1, x2: batch_x2})
                if j == 0:
                    tempEval = predOut
                else:
                    tempEval = np.concatenate((tempEval, predOut))
            predOut = np.argmax(tempEval, axis=1)
            tempLabels = np.argmax(testLabels, axis=1)
            print("Predictions: ", predOut.shape, "\n\n")
            print("Labels: ", tempLabels.shape, "\n\n")
            tn, fp, fn, tp = skm.confusion_matrix(tempLabels, predOut).ravel()
            print("\n\nConfusion Matrix:\n", skm.confusion_matrix(tempLabels, predOut), "\n\n")
            print("True Negative: ", tn)
            print("False Negative: ", fn)
            print("False Positive: ", fp)
            print("True Positive: ", tp)
            print("Accuracy: ", skm.accuracy_score(tempLabels, predOut), "%\n")
            print("Precision: ", skm.precision_score(tempLabels, predOut))
            print("Recall: ", skm.recall_score(tempLabels, predOut))
            print("F1 Score: ", skm.f1_score(tempLabels, predOut))

        save_path = saver.save(sess, modelFn)
        print("Model saved in file %s" % save_path)

####### LOAD THE PREPARED DATA FROM THE PICKLE FILES #######
try:
    trainBand1 = pickle.load(open("data/pickle/trainBand1.p", "rb"))
    trainBand2 = pickle.load(open("data/pickle/trainBand2.p", "rb"))
    testBand1 = pickle.load(open("data/pickle/testBand1.p", "rb"))
    testBand2 = pickle.load(open("data/pickle/testBand2.p", "rb"))
    trainLabels = pickle.load(open("data/pickle/trainLabels.p", "rb"))
    testLabels = pickle.load(open("data/pickle/testLabels.p", "rb"))
except:
    print("Problem loading the data from the pickle files... exiting application")
    exit(1)

###### RUN THE SESSION ########
print("\n\nStarting the TensorFlow session...\n\n")
train_network(x1, x2)
