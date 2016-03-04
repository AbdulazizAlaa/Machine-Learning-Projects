#!/usr/bin/python

from __future__ import division
from os import listdir
from os.path import isfile, join
import math

def compute_img_points(points, classNum):
    feature = []
    noseX = float(points[17].split(' ')[0])
    noseY = float(points[17].split(' ')[1])
    i = 3
    while i<23:
        point = points[i].split(' ')
        x = float(point[0])
        y = float(point[1])
        dist = abs((x-noseX)**2 + (y-noseY)**2)
        dist = math.sqrt(dist)
        feature.append(dist)
        i+=1
    feature.append(classNum)
    return feature

def train(files, path, featuresVector, classNum):
    for f in files:
        pointsfile = open(path+"/"+f)
        points = pointsfile.readlines()
        feature = compute_img_points(points, classNum)
        featuresVector.append(feature)
    return

def compute_img_dist(featuresList, testfeature):
    featureDist = []
    for img in featuresList:
        j = 0
        d = 0
        while j<20:
            d += (img[j]-testfeature[j])**2
            j+=1
        d = math.sqrt(abs(d))
        fd = [d, img[20]]
        featureDist.append(fd)
    featureDist.sort()
    # print featureDist
    return featureDist

def test_imgs(featuresVector, testfeatures, k):
    true = 0
    false = 0
    count = 1
    classes = ["Closing Eyes", "Looking Down", "Looking Front", "Looking Left"]
    for img in testfeatures:
        dist = compute_img_dist(featuresVector, img)
        j=0
        score = [0,0,0,0]
        while j<k:
             if dist[j][1] == 1:
                score[0]+=1
             if dist[j][1] == 2:
                score[1]+=1
             if dist[j][1] == 3:
                score[2]+=1
             if dist[j][1] == 4:
                score[3]+=1
             j+=1
        index = score.index(max(score))
        print " "
        print score
        print str(count) + "- Expected: " + classes[int(img[20])-1] + " -- Actual: " + classes[index]
        count+=1
        if int(img[20])-1 == index:
            true+=1
        else:
            false+=1
    print "\ntrue: " + str(true) + " out of " + str(true+false) + " ==> " + str(float(true/(true+false))*100) + "%"
    return

def get_files_list(path):
    files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".pts") ]
    return files

#trainingClosingEyes
trainingClosingEyespath = "Dataset/Training Dataset/ClosingEyes"
trainingClosingEyesfiles = get_files_list(trainingClosingEyespath)

#trainingLookingDown
trainingLookingDownpath = "Dataset/Training Dataset/LookingDown"
trainingLookingDownfiles = get_files_list(trainingLookingDownpath)

#trainingLookingFront
trainingLookingFrontpath = "Dataset/Training Dataset/LookingFront"
trainingLookingFrontfiles = get_files_list(trainingLookingFrontpath)

#trainingLookingLeft
trainingLookingLeftpath = "Dataset/Training Dataset/LookingLeft"
trainingLookingLeftfiles = get_files_list(trainingLookingLeftpath)

#testingClosingEyes
testingClosingEyespath = "Dataset/Testing Dataset/ClosingEyes"
testingClosingEyesfiles = get_files_list(testingClosingEyespath)

#testingLookingDown
testingLookingDownpath = "Dataset/Testing Dataset/LookingDown"
testingLookingDownfiles = get_files_list(testingLookingDownpath)

#testingLookingFront
testingLookingFrontpath = "Dataset/Testing Dataset/LookingFront"
testingLookingFrontfiles = get_files_list(testingLookingFrontpath)

#testingLookingLeft
testingLookingLeftpath = "Dataset/Testing Dataset/LookingLeft"
testingLookingLeftfiles = get_files_list(testingLookingLeftpath)


features = []
testfeatures = []
dist = 0

# Training
# ClosingEyes
train(trainingClosingEyesfiles, trainingClosingEyespath, features, 1)
# LookingDown
train(trainingLookingDownfiles, trainingLookingDownpath, features, 2)
# LookingFront
train(trainingLookingFrontfiles, trainingLookingFrontpath, features, 3)
# LookingLeft
train(trainingLookingLeftfiles, trainingLookingLeftpath, features, 4)

# testing
# ClosingEyes
train(testingClosingEyesfiles, testingClosingEyespath, testfeatures, 1)
# LookingDown
train(testingLookingDownfiles, testingLookingDownpath, testfeatures, 2)
# LookingFront
train(testingLookingFrontfiles, testingLookingFrontpath, testfeatures, 3)
# LookingLeft
train(testingLookingLeftfiles, testingLookingLeftpath, testfeatures, 4)

test_imgs(features, testfeatures, 15)
