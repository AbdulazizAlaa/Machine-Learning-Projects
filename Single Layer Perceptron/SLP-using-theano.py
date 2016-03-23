import theano.tensor as T
from theano import function
from theano import shared
from theano import config
import numpy as np
import matplotlib.pyplot as plt

# testX = np.load("testinput.npy")
# testt = np.load("testoutput.npy")
# trainX = np.load("input.npy")
# traint = np.load("output.npy")
#
# traint = np.reshape(traint, (traint.shape[0], 1))
# testt = np.reshape(testt, (testt.shape[0], 1))

trainX = np.matrix([[0, 1],
                   [0, 3],
                   [0, 10],
                   [-1, 0],
                   [-3, 0],
                   [-10, 0],
                   [0, -1],
                   [0, -3],
                   [0, -10],
                   [1, 0],
                   [3, 0],
                   [10, 0]])
traint = np.matrix([-1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1])

X = T.matrix('X')
t = T.matrix('t')

W = shared(value=np.random.randn(trainX.shape[1]+1), name='W')

predict = T.switch(T.dot(X, W) >= 0, 1, -1)
cost = T.sum(T.sub(predict, t)**2)
gradient = T.grad(T.sum(T.sub(T.dot(X, W), t)**2), W)

gradient_descient = function(inputs=[X, t], outputs=[cost, predict, gradient], updates=[(W, W-.001*gradient)])



trainX = np.append(np.ones((trainX.shape[0], 1)), trainX, axis=1)

for i in range(100):
    c = gradient_descient(trainX, traint)
    print c[0]

plt.scatter(trainX[:, 1], trainX[:, 2], marker='o', c=np.array(traint))
max = np.max( trainX[:, 1] )
min = np.min( trainX[:, 1] )
x1 = np.arange(min, max, 2)
x2 = - W.get_value()[0]/W.get_value()[2] - (W.get_value()[1]/W.get_value()[2]) * x1
plt.plot(x1, x2, 'k-')
plt.show()
