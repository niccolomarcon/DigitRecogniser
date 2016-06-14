import numpy as np
import matplotlib.pyplot as plt
import cPickle, gzip
from matplotlib import cm

# display functions
def displayDigit(digit):
    image = np.reshape(digit, (28, 28))
    plt.axis('off')
    plt.imshow(image, interpolation = "nearest", cmap = cm.Greys)

def prettyGuess(net, image):
    res = net.forward(image)
    guess = 0
    max = 0

    for i in range(10):
        if res[i] > max:
            max = res[i]
            guess = i

    return guess

def prettyLabel(label):
    for i in range(10):
        if label[i] == 1.:
            return i

def scaling(n, sets):
    digits, labels = sets

    digits = np.asarray(digits[:n])
    labels = np.asarray(labels[:n])
    labels = labels.reshape(n, 1)

    tmp = np.zeros((n, 10))
    for i in range(n):
        tmp[i][labels[i]] = 1.

    return digits, tmp

def load(N):
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    dtrain, ltrain = scaling(N, train_set)
    dtest, ltest   = scaling(N, test_set)

    return dtrain, ltrain, dtest, ltest
