# from theano import *
import numpy as n
# import theano.tensor as t

def GradientDescentTraining(X, T):
    lRate = .03
    
    tR = n.shape(T)[0]
    xR = n.shape(X)[0]
    i=0
    # Create random weights
    W = n.zeros(shape=(tR,xR))
    while (i<tR):
        j=0
        while (j<xR):
            W[i][j] = n.random.random()
            j+=1
        i+=1

    while (1):
        [c, G] = Cost(X, T, W)
        if (n.sum(G) == 0):
            break
        W = W - lRate * G
        #print c

    # print W*X
    # print W

    return

def Cost(X, T, W):
    m = n.shape(X)
    # Y => output
    Y = W*X
    # C => cost
    # C = 1/2 * sum[ (output - target)^2 ]
    d = Y-T
    C = n.sum(n.square(d))/2
    # G => gradient
    G = d * n.transpose(X)

    return [C, G]


X = n.matrix([[1, 2], [2, 1]])
T = n.matrix([[-1, 1]])

GradientDescentTraining(X,T)
