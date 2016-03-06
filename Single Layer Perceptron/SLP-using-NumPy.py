import numpy as n

def Predicting(X, T, W):
    dim = n.shape(T)
    m = n.shape(X)[0]
    # Appending a column of ones refers to X0 feature
    tempX = n.append(n.ones((m,1)), X, axis=1)

    # calculating predictations using the weights given by the gradient descent
    predict = W * n.transpose(tempX)
    predict = n.piecewise(predict, [predict < 0, predict >= 0], [-1, 1])

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

def GradientDescentTraining(X, T):
    # Learning Rate of Gradient Descent
    lRate = .01
    # size of target matrix
    sizeT = n.shape(T)
    # size of input matrix
    sizeX = n.shape(X)

    # Appending a column of ones refers to X0 feature
    X = n.append(n.ones((sizeX[0],1)), X, axis=1)
    # Create random weights with extra column for W0 weight
    W = n.zeros((sizeT[0],sizeX[1]+1))
    i=0
    while (i<sizeT[0]):
        j=0
        while (j<(sizeX[1]+1)):
            W[i][j] = n.random.random()
            j+=1
        i+=1
    # To detect Convergance make sure the change in weight is very small
    # deltaW = n.zeros((sizeT[0],sizeX[1]+1))

    while (1):
        [c, G] = Cost(X, T, W)
        if (n.sum(G) <= 0):
            break
        W = W - lRate * G

    return W

def Cost(X, T, W):
    m = n.shape(X)[0]
    # Y => output
    Y = W * n.transpose(X)
    # C => cost
    # C = 1/2 * sum[ (output - target)^2 ]
    d = Y-T
    c = n.sum(n.square(d))/(2*m)
    # G => gradient
    G = d * X

    return [c, G]


X = n.matrix([ [1, 2],
               [2, 1],
               [1,-1],
               [1, -2],
               [-3, -3],
               [-3, -1] ])

T = n.matrix([[-1, 1, 1, 1, -1, -1]])

W = GradientDescentTraining(X,T)

testingSetInput = n.matrix([ [50, 2],
                             [2, 50],
                             [1, 5],
                             [1, -20],
                             [-30, -31],
                             [-31, -30],
                             [-5, -4] ])
testingSetTarget = n.matrix([[1, -1, -1, 1, 1, -1, -1]])

Predicting(testingSetInput, testingSetTarget, W)
