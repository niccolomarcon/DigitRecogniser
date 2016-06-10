import random
import NeuralNetwork
import mnist
import matplotlib.pyplot as plt

print 'Loading the data set...'
number = 1000
digits_train, labels_train, digits_test, labels_test = mnist.load(number)
net = NeuralNetwork.NeuralNetwork(784, 15, 10)
print 'Data set loaded'

print '\nLearning...'
net.train(digits_train, labels_train, digits_test, labels_test)
print 'Learned'

# We can plot the training process
# plt.plot(net.TrainingPlot)
# plt.plot(net.OverfitPlot)
# plt.xlabel('Iterations')
# plt.ylabel('Error')
# plt.legend(['Training', 'Testing'])
# plt.show()
# plt.clf()

# Calculate the accuracy
right_train = 0.
right_test  = 0.

for i in range(number):
    if mnist.prettyGuess(net, digits_train[i]) == mnist.prettyLabel(labels_train[i]):
        right_train += 1.
    if mnist.prettyGuess(net, digits_test[i]) == mnist.prettyLabel(labels_test[i]):
        right_test += 1.

print '\nAccuracy:'
print '  train -> %.2f%%' % (right_train * 100. / number)
print '  test  -> %.2f%%' % (right_test * 100. / number)

# Test with one random number
n = random.randint(0, number - 1)
print '\nRandom test:'
print '  guess   %i' % mnist.prettyGuess(net, digits_test[n])
print '  correct %i' % mnist.prettyLabel(labels_test[n])
mnist.displayDigit(digits_test[n])
plt.show()
