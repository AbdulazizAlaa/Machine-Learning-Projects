import utils as du
import theano.tensor as T
from theano import function
from theano import shared
import numpy as np
import matplotlib.pyplot as plt


filename = "iris"

epochs = 5000
lRate = 0.01

# data = du.read_data("../data/"+filename+".data", 1, 100)
# du.save_data("../data/"+filename, data)
data = du.load_data("../data/"+filename+".pickle")

trainX = data[0][:, 0:2]
traint = data[1]

traint = np.where(traint == "Iris-setosa", 1, -1)

# trainX = np.matrix([[0, 1],
#                    [0, 3],
#                    [0, 10],
#                    [-1, 0],
#                    [-3, 0],
#                    [-10, 0],
#                    [0, -1],
#                    [0, -3],
#                    [0, -10],
#                    [1, 0],
#                    [3, 0],
#                    [10, 0]])
# traint = np.matrix([-1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1])

X = T.dmatrix('X')
t = T.dmatrix('t')

W = shared(value=np.random.randn(trainX.shape[1]+1, 1), name='W')

predict = T.switch(T.dot(X, W) >= 0, 1, -1)
cost = T.sum(T.sub(predict, t)**2)/(2*predict.shape[0])
gradient = T.grad(T.sum(T.sub(T.dot(X, W), t)**2)/(2*predict.shape[0]), W)

gradient_descient = function(inputs=[X, t], outputs=[cost, predict, gradient], updates=[(W, W-lRate*gradient)])

trainX = np.append(np.ones((trainX.shape[0], 1)), trainX, axis=1)

cost = []
for i in range(epochs):
    c = gradient_descient(trainX, traint)
    cost.append(c[0])
    print "epoch "+str(i)+": "+str(c[0])

plt.figure(1)
plt.subplot(211)
plt.scatter(trainX[:, 1], trainX[:, 2], marker='o', c=np.array(traint))
max = np.max( trainX[:, 1] )
min = np.min( trainX[:, 1] )
x1 = np.arange(min, max, 2)
x2 = - W.get_value()[0]/W.get_value()[2] - (W.get_value()[1]/W.get_value()[2]) * x1
plt.plot(x1, x2, 'k-')

plt.subplot(212)
plt.plot(range(0, epochs), cost, 'k-')

plt.show()
