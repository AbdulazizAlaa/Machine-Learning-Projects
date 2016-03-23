import pandas as pd
import numpy as np
import theano.tensor as T
from theano import function
from theano import shared
import matplotlib.pyplot as plt
from six.moves import cPickle

train_set = cPickle.load(open("train.pickle", "rb"))
test_set = cPickle.load(open("test.pickle", "rb"))

# input features
trainX = train_set[:, 1:]
trainX = (trainX - trainX.mean())/ trainX.std()

trainX = np.append(np.ones(shape=(np.shape(trainX)[0], 1)), trainX, axis=1)
# output
trainy = train_set[:, 0]

numLayers = 0
numFeatures = trainX.shape[1]
numOutputNeurons = trainy.shape[1]
numHiddenNeurons = 50
num_classes = 10
eta = 0.5
epochs = 200

# on hot
trainy_onhot = np.zeros((trainX.shape[0], num_classes))
trainy_onhot[np.arange(trainy.shape[0]), trainy] = 1

trainy = trainy_onhot

X = T.matrix('X')
t = T.matrix('t')

# Wji = shared(np.random.rand(numFeatures, numHiddenNeurons))
# Wkj = shared(np.random.rand(numHiddenNeurons, numOutputNeurons))
Wji = shared(np.random.rand(numHiddenNeurons, numFeatures))
Wkj = shared(np.random.rand(numOutputNeurons, numHiddenNeurons))
# print Wji.get_value()
# print Wkj.get_value()
# Aji = T.dot(X, Wji)
# Hji = T.nnet.sigmoid(Aji)
# Akj = T.dot(Hji, Wkj)
# y = T.nnet.softmax(Akj)
Aji = T.nnet.sigmoid(T.dot(Wji, X.T))
Akj = T.nnet.softmax(T.dot(Wkj, Aji))

# E = t
# E = T.sum(T.sub(t, y)**2)/2
E = T.sum(T.sub(t, Akj.T)**2)/2
# E = T.nnet.binary_crossentropy(Akj, t).mean()

gradji = T.grad(E, Wji)
gradkj = T.grad(E, Wkj)

updates = [(Wji, Wji-eta*gradji),
           (Wkj, Wkj-eta*gradkj)]

back_prop = function(inputs=[X, t], outputs=[E], updates=updates)
# back_prop = function(inputs=[X, t], outputs=[E, Akj])


# print back_prop(trainX, trainy)[1].shape
# print back_prop(trainX, trainy)
cost = []
for i in range(epochs):
    print "Epoch: " + str(i)
    # print back_prop(trainX, trainy)
    print "Cost:"
    print back_prop(trainX, trainy)[0]
    cost.append(back_prop(trainX, trainy)[0])
    print "Weights:"
    print Wji.get_value()

plt.plot(range(epochs), cost)
plt.show()

# print W.get_value()
# print len(W.get_value())
# print W.get_value()[0].shape
# print W.get_value()[1].shape
# print W.get_value()[2].shape
