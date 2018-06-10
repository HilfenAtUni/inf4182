import numpy as np
import mnist

# Load the raw MNIST
X_train, y_train = mnist.read(dataset='training')
X_test, y_test = mnist.read(dataset='testing')

# Subsample the data for more efficient code execution in this exercise
num_training = 6000
X_train = X_train[:num_training]
y_train = y_train[:num_training]

num_test = 500
X_test = X_test[:num_test]
y_test = y_test[:num_test]

# Reshape the image data into rows
# Datatype int allows you to subtract images (is otherwise uint8)
X_train = np.reshape(X_train, (X_train.shape[0], -1)).astype('int')
X_test = np.reshape(X_test, (X_test.shape[0], -1)).astype('int')

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


###############################################################################
#                                                                             #
#          Implement the k Nearest Neighbors algorithm here                   #
#                                                                             #
###############################################################################
class kNN:
    def train(self, X, y):
        """'train' the k-Nearest neighbor algorithm"""
        #TODO

    def classify(self, x, k):
        """evaluate the a single point or multiple points"""
        #TODO

#
# Test the kNN classifier
# (with 6000 train, 500 test, you should get an accuracy around 90%
#



###############################################################################
#                                                                             #
#                Implement Cross Validation here                              #
#                                                                             #
###############################################################################

def crossValidation(dataset, labels, n, classifier):
    """perform n-fold cross-validation using the given dataset and classifier"""
    #TODO

#
# Run the cross validation
#

#TODO


