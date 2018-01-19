###########################################################################################################
##
## data_manipulation.py
##
## @author Matthew Cline
## @version 20180106
##
## Description: Prepare the data for use in some TensorFlow Conv Nets. Save the objects to Pickle files for storage.
##
## Data: Data comes for this Kaggle Competition https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data
##
###########################################################################################################


import pandas as pd
import numpy as np
import os
import pickle
import scipy.signal as sig

######## PLACE TO SAVE THE MODEL AFTER TRAINING ########
dataDirectory = os.path.normpath('data/pickle')
if not os.path.exists(dataDirectory):
    os.makedirs(dataDirectory)

######## UTILITY FUNCTIONS ########
def splitData(data, trainingSplit=0.7):
    training, test = np.split(data, [int(data.shape[0] * trainingSplit)])
    return training, test


def shuffleData(data):
    data = data.reindex(np.random.permutation(data.index))
    data = data.reset_index(drop=True)
    return data


def standardizeFeatures(data):
    return (data - np.mean(data)) / np.std(data)
    
def scaleFeatures(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

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

def combineBands(data1, data2):
    assert len(data1) == len(data2)
    combo = []
    for i in range(len(data1)):
        temp = []
        assert len(data1[i]) == len(data2[i])
        for j in range(len(data1[i])):
            temp.append(data1[i][j])
            temp.append(data2[i][j])
        combo.append(np.array(temp))
    return np.array(combo)

def filterImages(data, width, height):
    for i in range(len(data)):
        temp = data[i]
        temp = np.reshape(temp, (width, height))
        temp = sig.medfilt2d(temp)
        temp = np.reshape(temp, (-1, width*height))
        data[i] = temp
    return data

####### READ IN THE DATA #######
print("\n\nReading the data from the file train.json ...\n\n")
tempData = pd.read_json('train.json')
print("Reading the data from the file test.json...\n\n")
compData = pd.read_json('test.json')

####### SET UP ONE HOT ENCODING #######
print("Encoding the labels with one hot encoding... \n\n")
tempData = encode_oneHot(tempData)

####### SHUFFLE THE DATA ########
print("Shuffling the data...\n\n")
tempData = shuffleData(tempData)

###### SPLIT THE DATA INTO TRAIN AND TEST #######
print("Splitting the data into training and validation...\n\n")
trainData, testData = splitData(tempData)

####### SPLIT THE IMAGES FROM THE LABELS #######
print("Splitting the images from the labels...\n\n")
trainLabels = np.array(trainData.iloc[:,5:])
testLabels = np.array(testData.iloc[:,5:])
trainBand1 = np.array(trainData.iloc[:,0])
trainBand2 = np.array(trainData.iloc[:,1])
testBand1 = np.array(testData.iloc[:,0])
testBand2 = np.array(testData.iloc[:,1])
compId = np.array(compData['id'])
compBand1 = np.array(compData.iloc[:,0])
compBand2 = np.array(compData.iloc[:,1])

####### CHANGE THE DATA STRUCTURE TO PLAY NICE WITH TF #######
print("Changing the data structure for TensorFlow...\n\n")
for i in range(len(trainBand1)):
    trainBand1[i] = np.array(trainBand1[i])
    trainBand2[i] = np.array(trainBand2[i])

trainBand1 = reorgImgs(trainBand1)
trainBand2 = reorgImgs(trainBand2)
    
for i in range(len(testBand1)):
    testBand1[i] = np.array(testBand1[i])
    testBand2[i] = np.array(testBand2[i])

testBand1 = reorgImgs(testBand1)
testBand2 = reorgImgs(testBand2)

for i in range(len(compBand1)):
    compBand1[i] = np.array(compBand1[i])
    compBand2[i] = np.array(compBand2[i])

compBand1 = reorgImgs(compBand1)
compBand2 = reorgImgs(compBand2)

####### STANDARDIZE THE IMAGE DATA ####### 
print("Standardizing the features...\n\n")
trainBand1 = standardizeFeatures(trainBand1)
trainBand2 = standardizeFeatures(trainBand2)
testBand1 = standardizeFeatures(testBand1)
testBand2 = standardizeFeatures(testBand2)
compBand1 = standardizeFeatures(compBand1)
compBand2 = standardizeFeatures(compBand2)

####### SCALING THE IMAGE DATA #######
print("Scaling the features...\n\n")
trainBand1 = scaleFeatures(trainBand1)
trainBand2 = scaleFeatures(trainBand2)
testBand1 = scaleFeatures(testBand1)
testBand2 = scaleFeatures(testBand2)
compBand1 = scaleFeatures(compBand1)
compBand2 = scaleFeatures(compBand2)

####### APPLY A MEDIAN FILTER TO REDUCE BACKSCATTER NOISE #######
print("Applying a median filter to the images to reduce noise...\n\n")
trainBand1_filtered = filterImages(trainBand1, 75, 75)
trainBand2_filtered = filterImages(trainBand2, 75, 75)
testBand1_filtered = filterImages(testBand1, 75, 75)
testBand2_filtered = filterImages(testBand2, 75, 75)
compBand1_filtered = filterImages(compBand1, 75, 75)
compBand2_filtered = filterImages(compBand2, 75, 75)

######## HORIZONTAL FLIP OF THE IMAGES #########
print("Performing a horizontal flip on the images...\n\n")
trainBand1_flipped = np.reshape(np.fliplr(np.reshape(trainBand1_filtered, (75,75))), (-1,5625))
trainBand2_flipped = np.reshape(np.fliplr(np.reshape(trainBand1_filtered, (75,75))), (-1,5625))
testBand1_flipped = np.reshape(np.fliplr(np.reshape(testBand1_filtered, (75,75))), (-1,5625))
testBand2_flipped = np.reshape(np.fliplr(np.reshape(testBand2_filtered, (75,75))), (-1,5625))
compBand1_flipped = np.reshape(np.fliplr(np.reshape(compBand1_filtered, (75,75))), (-1,5625))
compBand2_flipped = np.reshape(np.fliplr(np.reshape(compBand2_filtered, (75,75))), (-1,5625))

####### COMBINE THE BANDS INTO DUAL CHANNEL IMAGES #######
print("Combining band1 and band 2 into a single image...\n\n")
trainComboBand = combineBands(trainBand1, trainBand2)
testComboBand = combineBands(testBand1, testBand2)
compComboBand = combineBands(compBand1, compBand2)
print("Combined training bands shape: ", trainComboBand.shape, "\n\n")

######## DUMP ALL OF THE OBJECTS TO PICKLE FILES #######
print("Dumping all of the objects to Pickle files for later use.")
pickle.dump(testBand1, open(dataDirectory + "/testBand1.p", "wb"))
pickle.dump(testBand2, open(dataDirectory + "/testBand2.p", "wb"))
pickle.dump(testLabels, open(dataDirectory + "/testLabels.p", "wb"))
pickle.dump(trainBand1, open(dataDirectory + "/trainBand1.p", "wb"))
pickle.dump(trainBand2, open(dataDirectory + "/trainBand2.p", "wb"))
pickle.dump(trainLabels, open(dataDirectory + "/trainLabels.p", "wb"))
pickle.dump(compBand1, open(dataDirectory + "/compBand1.p", "wb"))
pickle.dump(compBand2, open(dataDirectory + "/compBand2.p", "wb"))

pickle.dump(compId, open(dataDirectory + "/compId.p", "wb"))
pickle.dump(trainComboBand, open(dataDirectory + "/trainComboBand.p", "wb"))
pickle.dump(testComboBand, open(dataDirectory + "/testComboBand.p", "wb"))
pickle.dump(compComboBand, open(dataDirectory + "/compComboBand.p", "wb"))

pickle.dump(trainBand1_filtered, open(dataDirectory + "/trainBand1_filtered.p", "wb"))
pickle.dump(trainBand2_filtered, open(dataDirectory + "/trainBand2_filtered.p", "wb"))
pickle.dump(testBand1_filtered, open(dataDirectory + "/testBand1_filtered.p", "wb"))
pickle.dump(testBand2_filtered, open(dataDirectory + "/testBand2_filtered.p", "wb"))
pickle.dump(compBand1_filtered, open(dataDirectory + "/compBand1_filtered.p", "wb"))
pickle.dump(compBand2_filtered, open(dataDirectory + "/compBand2_filtered.p", "wb"))

pickle.dump(trainBand1_flipped, open(dataDirectory + "/trainBand1_flipped.p", "wb"))
pickle.dump(trainBand1_flipped, open(dataDirectory + "/trainBand2_flipped.p", "wb"))
pickle.dump(testBand1_flipped, open(dataDirectory + "/testBand1_flipped.p", "wb"))
pickle.dump(testBand2_flipped, open(dataDirectory + "/testBand2_flipped.p", "wb"))
pickle.dump(compBand1_flipped, open(dataDirectory + "/compBand1_flipped.p", "wb"))
pickle.dump(compBand2_flipped, open(dataDirectory + "/compBand2_flipped.p", "wb"))
