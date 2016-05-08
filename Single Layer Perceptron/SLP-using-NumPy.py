import utils as du
import numpy as np
import matplotlib.pyplot as plt

def Predicting(X, T, W):
    dim = np.shape(T)
    # print dim
    # print T
    m = np.shape(X)[0]
    # Appending a column of ones refers to X0 feature
    tempX = np.append(np.ones((m,1)), X, axis=1)

    # calculating predictations using the weights given by the gradient descent
    predict = np.dot(tempX, np.transpose(W))
    predict = np.piecewise(predict, [predict < 0, predict >= 0], [-1, 1])

    # count for right predictions
    count = 0
    # loop counter for rows
    i=0
    while (i<dim[0]):
        # loop counter for columns
        j=0
        while (j<dim[1]):
            # if it is in same class then it is true then count+=1 else it is false
            if (predict[i, j] == T[i, j]):
                count += 1
                print X[j, :], " - predictation = ", predict[i, j], " - Target = ", T[i, j], " ===> true"
            else:
                print X[j, :], " - predictation = ", predict[i, j], " - Target = ", T[i, j], " ===> false"
            j+=1
        i+=1
    accuracy = ( count / float(m) ) * 100
    print "Accuracy = ", accuracy, " %"
    return

def GradientDescentTraining(X, T, epochs=50):
    # Learning Rate of Gradient Descent
    lRate = .01
    # size of target matrix
    sizeT = np.shape(T)
    # size of input matrix
    sizeX = np.shape(X)

    # Appending a column of ones refers to X0 feature
    X = np.append(np.ones((sizeX[0],1)), X, axis=1)
    # Create random weights with extra column for W0 weight
    W = np.zeros((1, sizeX[1]+1))

    j=0
    while (j<(sizeX[1]+1)):
        W[0, j] = np.random.random()
        j+=1

    cost = []
    for i in range(epochs):
        [c, G, d] = Cost(X, T, W)
        W[0, 0] = W[0, 0] - lRate * np.sum(d)
        W[0, 1:] = W[0, 1:] - lRate * G
        cost.append(c)
        print "epoch "+str(i)+": "+str(c)

    return [W, cost]

def Cost(X, T, W):
    m = np.shape(X)[0]
    # Y => output
    Y = np.dot(X, np.transpose(W))
    Y = np.where(Y >= 0, 1, -1)
    # C => cost
    # C = 1/2 * sum[ (output - target)^2 ]
    d = Y - T
    c = np.sum(np.square(d))/(2*m)
    # G => gradient
    G = np.dot(np.transpose(d), X[:, 1:])
    return [c, G, d]


epochs = 300
lRate = 0.01

filename = "iris"

# data = du.read_data("../data/"+filename+".data", 1, 100)
# du.save_data("../data/"+filename, data)
data = du.load_data("../data/"+filename+".pickle")

trainX = data[0][:, 0:2]
trainy = data[1]

trainy = np.where(trainy == "Iris-setosa", 1, -1)

# trainX = np.matrix([ [3,1,1],
#                      [4,0,1],
#                      [4,-1,1],
#                      [5,2,1],
#                      [5,3,1],
#                      [3,3,1],
#                      [2,0,1],
#                      [1,1,1] ])
# trainy = np.matrix([[]])

W, cost = GradientDescentTraining(trainX, trainy, epochs)

Predicting(trainX, trainy, W)

print "weights: " , W

plt.figure(1)
plt.subplot(211)

plt.scatter(trainX[:, 0], trainX[:, 1], marker='o', c=trainy)
max = np.max( trainX[:, 0] )
min = np.min( trainX[:, 0] )
x1 = np.arange(min, max, .1)
x2 = - W[0, 0]/W[0, 2] - (W[0, 1]/W[0, 2]) * x1
plt.plot(x1, x2, 'k-')

plt.subplot(212)
plt.plot(range(0, epochs), cost, 'k-')

plt.show()
