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
learningRate = 0.001
n_classes = 2
keepRate = 0.7
batchSize = 10
numEpochs = 10000
displayStep = 5

####### TENSORFLOW PLACEHOLDERS #######
x = tf.placeholder('float', [None, 5625])
y = tf.placeholder('float')

######## PLACE TO SAVE THE MODEL AFTER TRAINING ########
modelFn = os.path.normpath('models/tensorflow/iceberg_detector_single_band_network.ckpt')
if not os.path.exists(os.path.normpath('models/tensorflow')):
    os.makedirs('models/tensorflow')

######## UTILITY FUNCTIONS ########
def splitData(data, trainingSplit=0.7):
    training, test = np.split(data, [int(data.shape[0] * trainingSplit)])
    return training, test


def shuffleData(data):
    data = data.reindex(np.random.permutation(data.index))
    data = data.reset_index(drop=True)
    return data


def standardizeFeatures(data):
    for i in range(data.shape[0]):
        temp1 = np.array(data.iloc[i,0])
        temp2 = np.array(data.iloc[i,1])
        temp1 = (temp1 - np.mean(temp1)) / np.std(temp1)
        temp2 = (temp2 - np.mean(temp2)) / np.std(temp2)
        data.iloc[i,0] = temp1
        data.iloc[i,1] = temp2
    return data


def scaleFeatures(data):
    for i in range(data.shape[0]):
        temp1 = np.array(data.iloc[i,0])
        temp2 = np.array(data.iloc[i,1])
        temp1 = (temp1 - np.min(temp1)) / (np.max(temp1) - np.min(temp1))
        temp2 = (temp2 - np.min(temp2)) / (np.max(temp2) - np.min(temp2))
        data.iloc[i,0] = temp1
        data.iloc[i,1] = temp2
    return data


def encode_oneHot(data):
    data['not_iceberg'] = np.nan
    data['iceberg'] = np.nan
    for i in range(data.shape[0]):
        if data.iloc[i,4] == 1:
            data.iloc[i,5] = 0
            data.iloc[i,6] = 1
        else:
            data.iloc[i,5] = 1
            data.iloc[i,6] = 0
    return data

def reorgImgs(data):
    sample = data[0]
    for i in range(1,len(data)):
        sample = np.concatenate((sample, data[i]))
    sample =  np.reshape(sample, (-1,5625))
    return sample


def conv2d(x, W, b):
    temp = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b
    temp = tf.nn.leaky_relu(temp)
    temp = tf.nn.dropout(temp, keepRate)
    return temp


def convolutional_neural_network(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([3, 3, 1, 75])),
               'W_conv2': tf.Variable(tf.random_normal([4, 4, 75, 100])),
               'W_conv3': tf.Variable(tf.random_normal([4, 4, 100, 150])),
               'W_conv4': tf.Variable(tf.random_normal([4, 4, 150, 100])),
               'W_conv5': tf.Variable(tf.random_normal([4, 4, 100, 75])),
               'W_conv6': tf.Variable(tf.random_normal([3, 3, 75, 50])),
               'W_conv7': tf.Variable(tf.random_normal([3, 3, 50, 25])),
               'W_conv8': tf.Variable(tf.random_normal([3, 3, 25, 10])),
               'W_fc': tf.Variable(tf.random_normal([75 * 75 * 10, 1024])),
               'W_out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([75])),
              'b_conv2': tf.Variable(tf.random_normal([100])),
              'b_conv3': tf.Variable(tf.random_normal([150])),
              'b_conv4': tf.Variable(tf.random_normal([100])),
              'b_conv5': tf.Variable(tf.random_normal([75])),
              'b_conv6': tf.Variable(tf.random_normal([50])),
              'b_conv7': tf.Variable(tf.random_normal([25])),
              'b_conv8': tf.Variable(tf.random_normal([10])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'b_out': tf.Variable(tf.random_normal([n_classes]))}

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

    return output


def train_network(x):
    predictions = convolutional_neural_network(x)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predictions))

    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

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

            tempEval = []
            for j in range(int(len(testBand1) / batchSize)+1):
                batch_x = testBand1[j*batchSize:(j+1)*batchSize]
                predOut = predictions.eval(feed_dict={x: batch_x})
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

            tempEval = []
            for j in range(int(len(testBand1) / batchSize)+1):
                batch_x = testBand1[j*batchSize:(j+1)*batchSize]
                predOut = predictions.eval(feed_dict={x: batch_x})
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
                batch_x = testBand1[j*batchSize:(j+1)*batchSize]
                predOut = predictions.eval(feed_dict={x: batch_x})
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

# ####### READ IN THE DATA #######
# print("\n\nReading the data from the file train.json ...\n\n")
# tempData = pd.read_json('train.json')

# # ####### SCALE AND NORMALIZE THE DATA #######
# # print("Standardizing the features...\n\n")
# # tempData = standardizeFeatures(tempData)
# # print("Scaling the features...\n\n")
# # tempData = scaleFeatures(tempData)

# ####### SET UP ONE HOT ENCODING #######
# print("Encoding the labels with one hot encoding... \n\n")
# tempData = encode_oneHot(tempData)

# ####### SHUFFLE THE DATA ########
# print("Shuffling the data...\n\n")
# tempData = shuffleData(tempData)

# ###### SPLIT THE DATA INTO TRAIN AND TEST #######
# print("Splitting the data into training and validation...\n\n")
# trainData, testData = splitData(tempData)

# ####### SPLIT THE IMAGES FROM THE LABELS #######
# print("Splitting the images from the labels...\n\n")
# trainLabels = np.array(trainData.iloc[:,5:])
# testLabels = np.array(testData.iloc[:,5:])
# trainBand1 = np.array(trainData.iloc[:,0])
# trainBand2 = np.array(trainData.iloc[:,1])
# testBand1 = np.array(testData.iloc[:,0])
# testBand2 = np.array(testData.iloc[:,1])

# ####### CHANGE THE DATA STRUCTURE TO PLAY NICE WITH TF #######
# print("Changing the data structure for TensorFlow...\n\n")
# for i in range(len(trainBand1)):
#     trainBand1[i] = np.array(trainBand1[i])
#     trainBand2[i] = np.array(trainBand2[i])

# trainBand1 = reorgImgs(trainBand1)
# trainBand2 = reorgImgs(trainBand2)
    
# for i in range(len(testBand1)):
#     testBand1[i] = np.array(testBand1[i])
#     testBand2[i] = np.array(testBand2[i])

# testBand1 = reorgImgs(testBand1)
# testBand2 = reorgImgs(testBand2)


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
print("Starting the TensorFlow session...\n\n")
train_network(x)
