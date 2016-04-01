import pandas as pd
import numpy as np
import theano.tensor as T
from theano import function
from theano import shared
import matplotlib.pyplot as plt
from six.moves import cPickle

numHiddenNeurons1 = 300
numHiddenNeurons2 = 100
num_classes = 10
eta = 1
pre_training_epochs = 100
epochs = 500
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


# theano Functions

# weights of first hidden layer
Wji = shared(np.random.rand(numHiddenNeurons1, numFeatures)*0.01)
# weights of second hidden layer
Wkj = shared(np.random.rand(numHiddenNeurons2, numHiddenNeurons1)*0.01)
# weights of output layer in pre training when 1 hidden layer present
Wo = shared(np.random.rand(numOutputNeurons, numHiddenNeurons1)*0.01)
# weights of actual output layer
Wlk = shared(np.random.rand(numOutputNeurons, numHiddenNeurons2)*0.01)

# input values
X = T.dmatrix('X')
# output values
t = T.dmatrix('t')

# output of first hidden layer
Aji = T.nnet.sigmoid(T.dot(Wji, X.T))
# output of second hidden layer (prediction)
Akj = T.nnet.softmax(T.dot(Wo, Aji))

# error function of first pre training
E = T.mean(T.nnet.categorical_crossentropy(Akj.T, t))

# gradient of error with respect to weights of first hidden layer
gradji = T.grad(E, Wji)
# gradient of error with respect to weights of output layer in pre training when 1 hidden layer present
grado = T.grad(E, Wo)

updates = [(Wji, Wji-eta*gradji),
           (Wo, Wo-eta*grado)]

# pre training function
pre_training_first_stack = function(inputs=[X, t], outputs=[E], updates=updates)

# output of first hidden layer
Aji = T.nnet.sigmoid(T.dot(Wji, X.T))
# output of second hidden layer
Akj = T.nnet.sigmoid(T.dot(Wkj, Aji))
# output (prediction)
Alk = T.nnet.softmax(T.dot(Wlk, Akj))

# error function of second pre training
E = T.mean(T.nnet.categorical_crossentropy(Alk.T, t))

# gradient of error with respect to weights of second hidden layer
gradkj = T.grad(E, Wkj)
# gradient of error with respect to weights of actual output layer
gradlk = T.grad(E, Wlk)

updates = [(Wkj, Wkj-eta*gradkj),
           (Wlk, Wlk-eta*gradlk)]

# pre training function
pre_training_sec_stack = function(inputs=[X, t], outputs=[E], updates=updates)

# output of first hidden layer
Aji = T.nnet.sigmoid(T.dot(Wji, X.T))
# output of second hidden layer
Akj = T.nnet.sigmoid(T.dot(Wkj, Aji))
# output (prediction)
Alk = T.nnet.softmax(T.dot(Wlk, Akj))

# error function of training
E = T.mean(T.nnet.categorical_crossentropy(Alk.T, t))

# gradient of error with respect to weights of first hidden layer
gradji = T.grad(E, Wji)
# gradient of error with respect to weights of second hidden layer
gradkj = T.grad(E, Wkj)
# gradient of error with respect to weights of actual output layer
gradlk = T.grad(E, Wlk)

updates = [(Wji, Wji-eta*gradji),
           (Wkj, Wkj-eta*gradkj),
           (Wlk, Wlk-eta*gradlk)]

# training
training = function(inputs=[X, t], outputs=[E], updates=updates)


O = T.nnet.softmax(T.dot(Wlk, T.nnet.sigmoid(T.dot(Wkj, T.nnet.sigmoid(T.dot(Wji, X.T))))))
forward_prop = function(inputs=[X], outputs=[O])

costs = []
print "first Stack back Prop : "

for i in range(pre_training_epochs):
    print "Epoch: " + str(i)
    cost = pre_training_first_stack(trainX, trainy)[0]
    print "Cost:" + str(cost)
    costs.append(cost)

print "Second Stack back Prop : "

for i in range(pre_training_epochs):
    print "Epoch: " + str(i)
    cost = pre_training_sec_stack(trainX, trainy)[0]
    print "Cost:" + str(cost)
    costs.append(cost)

print "training back Prop : "

for i in range(epochs):
    print "Epoch: " + str(i)
    cost = training(trainX, trainy)[0]
    print "Cost:" + str(cost)
    costs.append(cost)

predictions = forward_prop(testX)
predictions = np.argmax(predictions, axis=1)[0]
testy = np.argmax(testy, axis=1)

num_true_predict = len(np.where(predictions == testy)[0])
num_records = len(testy)
score = ( float(num_true_predict)/num_records ) * 100

print "score: "+str(score) +" %"

plt.plot(range(pre_training_epochs*2+epochs), costs)
plt.show()
