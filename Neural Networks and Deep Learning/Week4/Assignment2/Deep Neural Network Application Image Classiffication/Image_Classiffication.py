# -------------------------------Packages--------------------------------
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v3 import *

# plt.rcParams['figure.figsize'] = (5.0, 4.0)
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmp'] = 'gray'

np.random.seed(1)

# -------------------------------Dataset--------------------------------
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# The following code will show you an image in the dataset. Feel free to change the
# index and re-run the cell multiple times to see other images.
# Example of picture
# index = 10
# plt.imshow(train_x_orig[index])
# print("y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]].decode("utf-8") + " picture.")

# Explore your dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

# -------------------------------Architecture of your model--------------------------------
# Two-layer neural network
# CONSTANTS DEFINING THE MODEL
# n_x = 12288
# n_h = 7
# n_y = 1
# layers_dims = (n_x, n_h, n_y)


# GRADED FUNCTION: two_layer_model


def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
       Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

       Arguments:
       X -- input data, of shape (n_x, number of examples)
       Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
       layers_dims -- dimensions of the layers (n_x, n_h, n_y)
       num_iterations -- number of iterations of the optimization loop
       learning_rate -- learning rate of the gradient descent update rule
       print_cost -- If set to True, this will print the cost every 100 iterations

       Returns:
       parameters -- a dictionary containing W1, W2, b1, and b2
    """
    np.random.seed(1)
    grads = {}
    costs = []
    m = X.shape[1]
    (n_x, n_h, n_y) = layers_dims

    parameters = initialize_parameters(n_x, n_h, n_y)

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    for i in range(0, num_iterations):
        A1, cache1 = linear_activation_forward(X, W1, b1, 'relu')
        A2, cache2 = linear_activation_forward(A1, W2, b2, 'sigmoid')

        cost = compute_cost(A2, Y)

        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, 'sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 'relu')

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        if print_cost and i % 100 == 0:
            print('Cost after interation{}: {}'.format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title('Learning rate =' + str(learning_rate))
    plt.show()

    return parameters


# Let's run the model

# parameters = two_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)
# predictions_train = predict(train_x, train_y, parameters)
# predictions_test = predict(test_x, test_y, parameters)

# L-layer Neural Network
# CONSTANTS
layers_dims = [12288, 20, 7, 5, 1]  # 4-layer model

# GRADED FUNCTION: L_layer_model

def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations = 3000, print_cost=False):
    """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

        Arguments:
        X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
    """
    np.random.seed(1)
    costs = []

    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print('Cost after iteration {}: {}'.format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title('learning rate = ' + str(learning_rate))
    plt.show()

    return parameters


# Let's run the model
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=3500, print_cost=True)
pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)


# -------------------------------Test with your own image--------------------------------
my_image = 'my_image.jpg'
my_label_y = [1]

fname = 'images/' + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((num_px * num_px * 3, 1))
my_image = my_image / 255
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" +
      classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")