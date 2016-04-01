import pandas as pd
import numpy as np
import cPickle as pickle


def read_data(filename, p, rows, outCols=0):
    dataframe = pd.read_csv(filename).iloc[:rows,:]
    if(outCols == 0):
        outCols = dataframe.shape[1]-1
    numRows = p*dataframe.shape[0]

    # dataframe.iloc[np.random.permutation(len(dataframe))]
    # np.random.shuffle(dataframe.iloc[:,:].values)

    indata = dataframe.iloc[:, :outCols].values
    outdata = dataframe.iloc[:, outCols].values

    train_X = indata[:numRows, :outCols]
    train_y = outdata[:numRows]
    train_y = train_y.reshape(train_y.shape[0], 1)

    test_X = indata[numRows:, :outCols]
    test_y = outdata[numRows:]
    test_y = test_y.reshape(test_y.shape[0], 1)

    return [train_X, train_y, test_X, test_y]



def save_data(filename, data):
    f = open(filename+".pickle", "w")

    pickle.dump(data, f)


def load_data(filename):
    f = open(filename, "r")

    data = pickle.load(f)

    return data
