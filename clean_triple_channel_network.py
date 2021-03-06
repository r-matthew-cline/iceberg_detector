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
import pickle

import tensorflow as tf



####### HYPER PARAMS ########
learningRate = 0.00001
beta = 0.01
n_classes = 2
keepRate = 0.95
if sys.argv[1] == 'test':
    keepRate = 1.0
batchSize = 48
numEpochs = 10000
displayStep = 5

####### TENSORFLOW PLACEHOLDERS #######
x = tf.placeholder('float', [None, 16875])
y = tf.placeholder('float')
weights = {'W_conv1': tf.Variable(tf.truncated_normal([3, 3, 3, 75], stddev=0.1)),
               'W_conv2': tf.Variable(tf.truncated_normal([4, 4, 75, 100], stddev=0.1)),
               'W_conv3': tf.Variable(tf.truncated_normal([4, 4, 100, 150], stddev=0.1)),
               'W_conv4': tf.Variable(tf.truncated_normal([4, 4, 150, 100], stddev=0.1)),
               'W_conv5': tf.Variable(tf.truncated_normal([4, 4, 100, 75], stddev=0.1)),
               'W_conv6': tf.Variable(tf.truncated_normal([3, 3, 75, 50], stddev=0.1)),
               'W_conv7': tf.Variable(tf.truncated_normal([3, 3, 50, 25], stddev=0.1)),
               'W_conv8': tf.Variable(tf.truncated_normal([3, 3, 25, 10], stddev=0.1)),
               'W_fc': tf.Variable(tf.truncated_normal([75 * 75 * 10, 1024], stddev=0.1)),
               'W_fc2': tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.1)),
               'W_fc3': tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.1)),
               'W_fc4': tf.Variable(tf.truncated_normal([1024, 512], stddev=0.1)),
               'W_fc5': tf.Variable(tf.truncated_normal([512, 256], stddev=0.1)),
               'W_fc6': tf.Variable(tf.truncated_normal([256, 32], stddev=0.1)),
               'W_out': tf.Variable(tf.truncated_normal([32, n_classes], stddev=0.1))}

biases = {'b_conv1': tf.Variable(tf.zeros([75])),
              'b_conv2': tf.Variable(tf.zeros([100])),
              'b_conv3': tf.Variable(tf.zeros([150])),
              'b_conv4': tf.Variable(tf.zeros([100])),
              'b_conv5': tf.Variable(tf.zeros([75])),
              'b_conv6': tf.Variable(tf.zeros([50])),
              'b_conv7': tf.Variable(tf.zeros([25])),
              'b_conv8': tf.Variable(tf.zeros([10])),
              'b_fc': tf.Variable(tf.zeros([1024])),
              'b_fc2': tf.Variable(tf.zeros([1024])),
              'b_fc3': tf.Variable(tf.zeros([1024])),
              'b_fc4': tf.Variable(tf.zeros([512])),
              'b_fc5': tf.Variable(tf.zeros([256])),
              'b_fc6': tf.Variable(tf.zeros([32])),
              'b_out': tf.Variable(tf.zeros([n_classes]))}

######## PLACE TO SAVE THE MODEL AFTER TRAINING ########
modelFn = os.path.normpath('models/tensorflow/clean_triple_channel_network3.ckpt')
if not os.path.exists(os.path.normpath('models/tensorflow')):
    os.makedirs('models/tensorflow')

####### SET UP LOGGING DIRECTORY FOR TENSORBOARD #######
logFn = os.path.normpath('models/tensorflow/logs/clean_triple_channel_network3')
if not os.path.exists(os.path.normpath('models/tensorflow/logs')):
    os.makedirs('models/tensorflow/logs')


####### UTILITY FUNCTIONS #######
def getProbs(predictions):
    probs = predictions
    for i in range(len(predictions)):
        temp = predictions[i]
        temp = np.exp(temp) / np.sum(np.exp(temp), axis=0)
        probs[i] = temp
    return probs

def conv2d(x, W, b):
    temp = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b
    temp = tf.nn.leaky_relu(temp)
    temp = tf.nn.dropout(temp, keepRate)
    return temp


def convolutional_neural_network(x, weights, biases):
    
    x = tf.reshape(x, shape=[-1, 75, 75, 3])

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

    fc2 = tf.nn.leaky_relu(tf.matmul(fc, weights['W_fc2']) + biases['b_fc2'])
    fc2 = tf.nn.dropout(fc2, keepRate)

    fc3 = tf.nn.leaky_relu(tf.matmul(fc2, weights['W_fc3']) + biases['b_fc3'])
    fc3 = tf.nn.dropout(fc3, keepRate)

    fc4 = tf.nn.leaky_relu(tf.matmul(fc3, weights['W_fc4']) + biases['b_fc4'])
    fc4 = tf.nn.dropout(fc4, keepRate)

    fc5 = tf.nn.leaky_relu(tf.matmul(fc4, weights['W_fc5']) + biases['b_fc5'])
    fc5 = tf.nn.dropout(fc5, keepRate)

    fc6 = tf.nn.leaky_relu(tf.matmul(fc5, weights['W_fc6']) + biases['b_fc6'])
    fc6 = tf.nn.dropout(fc6, keepRate)

    output = tf.matmul(fc6, weights['W_out']) + biases['b_out']
    
    if(sys.argv[1] == 'test'):
        return tf.nn.softmax(output)

    return output


def train_network(x):
    predictions = convolutional_neural_network(x, weights, biases)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predictions))
    # regularizer = tf.nn.l2_loss(weights['W_conv1']) + tf.nn.l2_loss(weights['W_conv2']) + tf.nn.l2_loss(weights['W_conv3']) + tf.nn.l2_loss(weights['W_conv4']) + tf.nn.l2_loss(weights['W_conv5']) + tf.nn.l2_loss(weights['W_conv6']) + tf.nn.l2_loss(weights['W_conv7']) + tf.nn.l2_loss(weights['W_conv8']) + tf.nn.l2_loss(weights['W_fc']) + tf.nn.l2_loss(weights['W_out'])
    # cost = tf.reduce_mean(cost + beta * regularizer)

    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ####### INIT LOGGING TO TENSORBOARD #######
        writer = tf.summary.FileWriter(logFn)
        writer.add_graph(sess.graph)

        print("\n\n")

        if sys.argv[1] == 'train':
            print("Beginning the training of the model...\n\n")
            for epoch in range(numEpochs + 1):
                epochLoss = 0
                for j in range(int(len(trainBand1) / batchSize)):
                    epoch_x = trainBand1[j*batchSize:(j+1)*batchSize]
                    epoch_y = trainLabels[j*batchSize:(j+1)*batchSize]
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epochLoss += c
                if epoch % displayStep == 0:
                    print("Epoch ", epoch + 1, ' completed out of ', numEpochs, ' with loss: ', epochLoss)
                    save_path = saver.save(sess, modelFn)
                    print("Model saved in file: %s" % save_path)

            print("Training Finished Successfully...")
            save_path = saver.save(sess, modelFn)
            print("Model saved in file: %s" % save_path)

        ####### Continue Training from a saved model #######
        elif sys.argv[1] == 'continue':
            print("Loading the model from storage to continue training...\n\n")
            saver.restore(sess, modelFn)
            for epoch in range(numEpochs + 1):
                epochLoss = 0
                for j in range(int(len(trainBand1) / batchSize)):
                    epoch_x = trainBand1[j*batchSize:(j+1)*batchSize]
                    epoch_y = trainLabels[j*batchSize:(j+1)*batchSize]
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epochLoss += c
                if epoch % displayStep == 0:
                    print("Epoch ", epoch + 1, ' completed out of ', numEpochs, ' with loss: ', epochLoss)
                    save_path = saver.save(sess, modelFn)
                    print("Model saved in file: %s" % save_path)

            print("Training Finished Successfully...")
            save_path = saver.save(sess, modelFn)
            print("Model saved in file: %s" % save_path)

        ####### Run only the test data throught the model #######
        else:
            print("Loading the model from storage...\n\n")
            saver.restore(sess, modelFn)
            tempEval = []
            for j in range(int(len(testBand1) / batchSize)+1):
                batch_x = testBand1[j*batchSize:(j+1)*batchSize]
                predOut = predictions.eval(feed_dict={x: batch_x})
                if j == 0:
                    tempEval = predOut
                else:
                    tempEval = np.concatenate((tempEval, predOut))
            predOut = np.argmax(tempEval, axis=1)
            probs = tempEval[:,1]
            lowThresh = 0.08
            highThresh = 0.92
            probs[probs < lowThresh] = lowThresh
            probs[probs > highThresh] = highThresh
            if len(sys.argv) > 2:
                if sys.argv[2] == 'hist':
                    plt.figure()
                    plt.hist(probs)
                    plt.show()
            tempLabels = np.argmax(testLabels, axis=1)
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
            print("Log Loss: ", skm.log_loss(tempLabels, probs, eps=1e-15), "\n\n")
            

        save_path = saver.save(sess, modelFn)
        print("Model saved in file %s" % save_path)

####### LOAD THE PREPARED DATA FROM THE PICKLE FILES #######
try:
    print("Loading the training images...")
    trainBand1 = pickle.load(open("data/pickle/cleanTripleTrain.p", "rb"))
    print("Loading the test imgages...")
    testBand1 = pickle.load(open("data/pickle/cleanTripleTest.p", "rb"))
    print("Loading the training labels...")
    trainLabels = pickle.load(open("data/picklecleanTripleTrainLabels.p", "rb"))
    print("Loading the test labels...")
    testLabels = pickle.load(open("data/pickle/testLabels.p", "rb"))
except:
    print("Problem loading the data from the pickle files... exiting application")
    exit(1)

###### RUN THE SESSION ########
print("\n\nStarting the TensorFlow session...\n\n")
train_network(x)
