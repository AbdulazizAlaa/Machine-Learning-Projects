import pandas as pd
import numpy as np
import theano.tensor as T
from theano import function
from theano import shared
import matplotlib.pyplot as plt
from six.moves import cPickle

numHiddenNeurons = 500
num_classes = 10
eta = 0.5
epochs = 1000
cross_val = .3


train_set = cPickle.load(open("../data/minst-train.pickle", "rb"))
test_set = cPickle.load(open("../data/minst-test.pickle", "rb"))

# Testing input features
testX = test_set[:, 1:]
testX = (testX - testX.mean())/ testX.std()

testX = np.append(np.ones(shape=(np.shape(testX)[0], 1)), testX, axis=1)
# output
testy = test_set[:, 0]
testy_onhot = np.zeros((testX.shape[0], num_classes))
testy_onhot[np.arange(testy.shape[0]), testy] = 1

testy = testy_onhot

# Training input features
trainX = train_set[:, 1:]
trainX = (trainX - trainX.mean())/ trainX.std()

trainX = np.append(np.ones(shape=(np.shape(trainX)[0], 1)), trainX, axis=1)
# output
trainy = train_set[:, 0]

trainy_onhot = np.zeros((trainX.shape[0], num_classes))
trainy_onhot[np.arange(trainy.shape[0]), trainy] = 1

trainy = trainy_onhot

numFeatures = trainX.shape[1]
numOutputNeurons = trainy.shape[1]

Wji = shared(np.random.rand(numHiddenNeurons, numFeatures))
Wkj = shared(np.random.rand(numOutputNeurons, numHiddenNeurons))

X = T.dmatrix('X')
t = T.dmatrix('t')

Aji = T.nnet.sigmoid(T.dot(Wji, X.T))
Akj = T.nnet.softmax(T.dot(Wkj, Aji))

# E = T.sum(T.sub(t, Akj.T)**2)/2
E = T.mean(T.nnet.categorical_crossentropy(Akj.T, t))

gradji = T.grad(E, Wji)
gradkj = T.grad(E, Wkj)

updates = [(Wji, Wji-eta*gradji),
           (Wkj, Wkj-eta*gradkj)]

back_prop = function(inputs=[X, t], outputs=[E], updates=updates)
# back_prop = function(inputs=[X, t], outputs=[E, Akj])

O = T.nnet.softmax(T.dot(Wkj, T.nnet.sigmoid(T.dot(Wji, X.T))))
forward_prop = function(inputs=[X], outputs=[O])

costs = []
for i in range(epochs):
    print "Epoch: " + str(i)
    cost = back_prop(trainX, trainy)[0]
    print "Cost:" + str(cost)
    costs.append(cost)

predictions = forward_prop(testX)
predictions = np.argmax(predictions, axis=1)[0]
testy = np.argmax(testy, axis=1)

num_true_predict = len(np.where(predictions == testy)[0])
num_records = len(testy)
score = ( float(num_true_predict)/num_records ) * 100

print "score: "+str(score) +" %"

plt.plot(range(epochs), costs)
plt.show()
