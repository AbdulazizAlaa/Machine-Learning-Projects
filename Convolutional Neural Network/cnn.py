import numpy as np
import theano.tensor as T
import theano.tensor.signal.pool as pool
from theano import function
from theano import shared
from six.moves import cPickle
import pylab
from PIL import Image

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def show_image(img, filtered):
    pylab.figure(1)
    pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
    pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered[0], cmap='gray')
    pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(filtered[1], cmap='gray')
    pylab.figure(2)
    pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(filtered[2], cmap='gray')
    pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered[3], cmap='gray')
    pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(filtered[4], cmap='gray')

    # pylab.show()

def get_accuracy(predict, actual):
    p = np.argmax(predict, axis=1)
    num_good = len(np.where(p==actual)[0])
    acc = (float(num_good)/np.shape(actual)[0])*100
    return acc

class Convultional_layer:

    def __init__(self, X, NUM_FILTERS, NUM_CHANELS, FILTER_DIMS, STRIDE, pool_shape, factor):
        self.kernals = shared(
            np.random.randn(
                NUM_FILTERS,
                NUM_CHANELS,
                FILTER_DIMS[0],
                FILTER_DIMS[1]
            ).astype(X.dtype)*factor,
            name='kernal'
        )

        self.b = shared(
            np.zeros(NUM_FILTERS).astype(X.dtype),
            name='b'
        )

        # OLD Gradients
        self.old_kernals = shared(
            np.zeros(
                (
                    NUM_FILTERS,
                    NUM_CHANELS,
                    FILTER_DIMS[0],
                    FILTER_DIMS[1]
                )
            ).astype(X.dtype),
            name='kernal'
        )

        self.old_b = shared(
            np.zeros(NUM_FILTERS).astype(X.dtype),
            name='b'
        )

        self.output = T.nnet.conv2d(
            input=X,
            filters=self.kernals,
            subsample=(STRIDE, STRIDE),
            border_mode='half'
        )

        self.output = T.nnet.relu( self.output + self.b.dimshuffle('x', 0, 'x', 'x') )

        self.output = pool.pool_2d(
            self.output,
            pool_shape,
            ignore_border='false'
        )

        self.params = [self.kernals, self.b]
        self.old_params = [self.old_kernals, self.old_b]

class FullyConnectedLayer:

    def __init__(self, X, num_hidden_neurons, num_outputs, num_inputs, factor):
        self.num_inputs = num_inputs

        self.NUM_HIDDEN_NEURONS = num_hidden_neurons
        self.Wji = shared(
            np.random.randn(
                self.NUM_HIDDEN_NEURONS,
                self.num_inputs
            ).astype(X.dtype)*factor
        )

        self.bji = shared(
            np.zeros(
                self.NUM_HIDDEN_NEURONS
            ).astype(X.dtype)
        )

        # Old gradients
        self.old_Wji = shared(
            np.zeros(
                (
                    self.NUM_HIDDEN_NEURONS,
                    self.num_inputs
                )
            ).astype(X.dtype)
        )

        self.old_bji = shared(
            np.zeros(
                self.NUM_HIDDEN_NEURONS
            ).astype(X.dtype)
        )

        self.num_outputs = num_outputs
        self.Wkj = shared(
            np.random.randn(
                self.num_outputs,
                self.NUM_HIDDEN_NEURONS
            ).astype(X.dtype)*factor
        )

        self.bkj = shared(
            np.zeros(
                self.num_outputs
            ).astype(X.dtype)*factor
        )

        # Old gradients
        self.old_Wkj = shared(
            np.zeros(
                (
                    self.num_outputs,
                    self.NUM_HIDDEN_NEURONS
                )
            ).astype(X.dtype)
        )

        self.old_bkj = shared(
            np.zeros(
                self.num_outputs
            ).astype(X.dtype)
        )

        Netji = T.dot(X, self.Wji.T)
        Aji = T.nnet.sigmoid(Netji + self.bji)

        Netkj = T.dot(Aji, self.Wkj.T)
        Akj = T.nnet.softmax(Netkj + self.bkj)

        self.output = Akj

        self.params = [self.Wkj, self.bkj, self.Wji, self.bji]
        self.old_params = [self.old_Wkj, self.old_bkj, self.old_Wji, self.old_bji]

# img = mpimg.imread('3wolfmoon.jpg')
# img = Image.open(open('3wolfmoon.jpg'))

NUM_FILTERS = 16
FILTER_SIZE = (5, 5)

NUM_FILTERS_2 = 20
FILTER_SIZE_2 = (5, 5)

NUM_FILTERS_3 = 20
FILTER_SIZE_3 = (5, 5)

STRIDE = 1
POOL_SHAPE = (2, 2)
NUM_HIDDEN_NEURONS = 1000

epochs = 15000
num_samples = 0
t_num_samples = 0
mini_batch = 200
eta = 0.5
factor = 0.001
momentum = 0.00

# Cifar-10 dataset

IMG_DIMS = (32, 32, 3)
IMG_DIMS_2 = (IMG_DIMS[0]/(POOL_SHAPE[0]*STRIDE), IMG_DIMS[1]/(POOL_SHAPE[1]*STRIDE), NUM_FILTERS)
IMG_DIMS_3 = (IMG_DIMS_2[0]/(POOL_SHAPE[0]*STRIDE), IMG_DIMS_2[1]/(POOL_SHAPE[1]*STRIDE), NUM_FILTERS_2)
IMG_DIMS_4 = (IMG_DIMS_3[0]/(POOL_SHAPE[0]*STRIDE), IMG_DIMS_3[1]/(POOL_SHAPE[1]*STRIDE), NUM_FILTERS_3)
NUM_OUTPUTS = 10
total_num_pixels = IMG_DIMS_2[0] * IMG_DIMS_2[1] * IMG_DIMS_2[2]
total_num_pixels_2 = IMG_DIMS_3[0] * IMG_DIMS_3[1] * IMG_DIMS_3[2]
total_num_pixels_3 = IMG_DIMS_4[0] * IMG_DIMS_4[1] * IMG_DIMS_4[2]

batch_1 = unpickle("../data/cifar-10-batches-py/data_batch_1")
batch_2 = unpickle("../data/cifar-10-batches-py/data_batch_2")
batch_3 = unpickle("../data/cifar-10-batches-py/data_batch_3")
batch_4 = unpickle("../data/cifar-10-batches-py/data_batch_4")
batch_5 = unpickle("../data/cifar-10-batches-py/data_batch_5")

data = []
data.append(batch_1['data'])
data.append(batch_2['data'])
data.append(batch_3['data'])
data.append(batch_4['data'])
data.append(batch_5['data'])
data = np.concatenate(data)

labels = []
labels.append(batch_1['labels'])
labels.append(batch_2['labels'])
labels.append(batch_3['labels'])
labels.append(batch_4['labels'])
labels.append(batch_5['labels'])
labels = np.concatenate(labels)

test_data = unpickle("../data/cifar-10-batches-py/test_batch")

num_samples = np.shape(data)[0]
t_num_samples = np.shape(test_data['data'])[0]

# training data
one_hot = np.zeros( (num_samples, NUM_OUTPUTS) )
one_hot[range(num_samples), labels] = 1
target = one_hot.astype('float32')

img = data.reshape((num_samples, IMG_DIMS[0], IMG_DIMS[1], IMG_DIMS[2]), order='F')

img = img.astype('float32') / 256

data_mean = data.mean()
# data_std = data.std()

data = (data - data_mean)
# data = data / data_std
########
# cov = np.dot(data.T, data) / np.shape(data)[0]

# U,S,V = np.linalg.svd(cov)

# Xrot = np.dot(data, U)

# Xrot_reduced = np.dot(data, U[:, :100])
# Xwhite = Xrot / np.sqrt(S + 1e-1)

# data = Xrot
########
img_ = data.reshape((-1, IMG_DIMS[2], IMG_DIMS[0], IMG_DIMS[0])).transpose(0, 2, 3, 1).reshape((-1, IMG_DIMS[2], IMG_DIMS[0], IMG_DIMS[1]))

# testing data
t_data = test_data['data']

t_img = t_data.reshape((t_num_samples, IMG_DIMS[0], IMG_DIMS[1], IMG_DIMS[2]), order='F')

t_img = t_img.astype('float32') / 256

t_data = (t_data - data_mean)
# t_data = t_data / data_std
########
# t_cov = np.dot(t_data.T, t_data) / np.shape(t_data)[0]
#
# t_U,t_S,t_V = np.linalg.svd(t_cov)

# t_Xrot = np.dot(t_data, U)

# t_Xwhite = t_Xrot / np.sqrt(t_S + 1e-1)

# t_data = t_Xrot
########
t_img_ = t_data.reshape((-1, IMG_DIMS[2], IMG_DIMS[0], IMG_DIMS[0])).transpose(0, 2, 3, 1).reshape((-1, IMG_DIMS[2], IMG_DIMS[0], IMG_DIMS[1]))

t_target = test_data['labels']

# ###########

# Minst dataset

# IMG_DIMS = (28, 28, 1)
# IMG_DIMS_2 = (IMG_DIMS[0]/(POOL_SHAPE[0]*STRIDE), IMG_DIMS[1]/(POOL_SHAPE[1]*STRIDE), NUM_FILTERS)
# IMG_DIMS_3 = (IMG_DIMS_2[0]/(POOL_SHAPE[0]*STRIDE), IMG_DIMS_2[1]/(POOL_SHAPE[1]*STRIDE), NUM_FILTERS_2)
# IMG_DIMS_4 = (IMG_DIMS_3[0]/(POOL_SHAPE[0]*STRIDE), IMG_DIMS_3[1]/(POOL_SHAPE[1]*STRIDE), NUM_FILTERS_3)
# NUM_OUTPUTS = 10
# total_num_pixels = IMG_DIMS_2[0] * IMG_DIMS_2[1] * IMG_DIMS_2[2]
# total_num_pixels_2 = IMG_DIMS_3[0] * IMG_DIMS_3[1] * IMG_DIMS_3[2]
# total_num_pixels_3 = IMG_DIMS_4[0] * IMG_DIMS_4[1] * IMG_DIMS_4[2]
#
# train_data = unpickle("../data/minst-train.pickle")
# test_data = unpickle("../data/minst-test.pickle")
#
# num_samples = np.shape(train_data)[0]
# t_num_samples = np.shape(test_data)[0]
#
# # testing data
# t_data = test_data[:, 1:]
#
# t_img = t_data.reshape((t_num_samples, IMG_DIMS[0], IMG_DIMS[1]), order='F')
#
# t_data = (t_data - t_data.mean())
#
# t_img_ = t_data.reshape((t_num_samples, IMG_DIMS[2], IMG_DIMS[0], IMG_DIMS[1]))
#
# t_target = test_data[:, 0]
#
# # training data
# data = train_data[:, 1:]
#
# img = data.reshape((num_samples, IMG_DIMS[0], IMG_DIMS[1]), order='F')
#
# data = (data - data.mean())
#
# img_ = data.reshape((num_samples, IMG_DIMS[2], IMG_DIMS[0], IMG_DIMS[1]))
#
# labels = train_data[:, 0]
# one_hot = np.zeros( (num_samples, NUM_OUTPUTS) )
# one_hot[range(num_samples), labels] = 1
# target = one_hot

# ###########

# Theano convolution_nnet Function

X = T.tensor4('X')
t = T.imatrix('t')
batch_size = T.iscalar('batch_size')

params = []
old_params = []

conv_layer = Convultional_layer(X=X, NUM_FILTERS=NUM_FILTERS, NUM_CHANELS=IMG_DIMS[2], FILTER_DIMS=FILTER_SIZE, STRIDE=STRIDE, pool_shape=POOL_SHAPE, factor=factor)
feature_maps = conv_layer.output
params.append(conv_layer.params)
old_params.append(conv_layer.old_params)

conv_layer_2 = Convultional_layer(X=feature_maps, NUM_FILTERS=NUM_FILTERS_2, NUM_CHANELS=IMG_DIMS_2[2], FILTER_DIMS=FILTER_SIZE_2, STRIDE=STRIDE, pool_shape=POOL_SHAPE, factor=factor)
feature_maps_2 = conv_layer_2.output
params.append(conv_layer_2.params)
old_params.append(conv_layer_2.old_params)

conv_layer_3 = Convultional_layer(X=feature_maps_2, NUM_FILTERS=NUM_FILTERS_3, NUM_CHANELS=IMG_DIMS_3[2], FILTER_DIMS=FILTER_SIZE_3, STRIDE=STRIDE, pool_shape=POOL_SHAPE, factor=factor)
feature_maps_3 = conv_layer_3.output
params.append(conv_layer_3.params)
old_params.append(conv_layer_3.old_params)

reshaped_image = T.reshape(feature_maps_3, (batch_size, total_num_pixels_3))
FC = FullyConnectedLayer(X=reshaped_image, num_hidden_neurons=NUM_HIDDEN_NEURONS, num_outputs=NUM_OUTPUTS, num_inputs=total_num_pixels_3, factor=factor)
predict = FC.output
params.append(FC.params)
old_params.append(FC.old_params)

params = np.concatenate(params)
old_params = np.concatenate(old_params)

cost = T.mean(T.nnet.categorical_crossentropy(predict, t))

gradients = []

for param in params:
    gradients.append(T.grad(cost, param))

updates = []

for param, grad, old_param in zip(params, gradients, old_params):
    updates.append((param, param-eta*grad+momentum*old_param))
for old_param, grad in zip(old_params, gradients):
    updates.append((old_param, grad))


convolution_nnet = function(inputs=[X, t, batch_size], outputs=[feature_maps, predict, cost], updates = updates, allow_input_downcast=True, on_unused_input='ignore')

forward_prop = function(inputs=[X, batch_size], outputs=[predict], allow_input_downcast=True, on_unused_input='ignore')
# ###########


cost = []
accu = []
lastAccu = 0
for i in range(epochs):
    index = np.random.randint(0, num_samples - mini_batch)

    images = img[index:index+mini_batch]
    mini_batch_images = img_[index:index+mini_batch]
    mini_batch_target = target[index:index+mini_batch]

    feature_maps, predict, c = convolution_nnet(mini_batch_images, mini_batch_target, mini_batch)

    cost.append(c)

    print "Epoch "+str(i)+":"
    print "Cost: "+str(c)
    print "Last Accuracy: "+str(lastAccu)
    if(i%500 == 0 and i>0):
        t_predict = forward_prop(t_img_, np.shape(t_img_)[0])
        accuracy = get_accuracy(t_predict[0], t_target)
        lastAccu = accuracy
        accu.append(accuracy)
        print "Accuracy : "+str(accuracy)+"%"

        # input("insert any letter")

    if(i == epochs-1):
        print np.shape(feature_maps)
        t_predict = forward_prop(t_img_, np.shape(t_img_)[0])
        print "Accuracy : "+str(get_accuracy(t_predict[0], t_target))+"%"
        show_image(images[2], feature_maps[2])
        pylab.figure('3')
        pylab.plot(range(epochs), cost, 'k-')
        pylab.figure('4')
        pylab.plot(range(len(accu)), accu, 'k-')
        pylab.show()
