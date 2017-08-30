# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 08:58:37 2017

@author: wdsxsx
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
#from lr_utils import load_dataset


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = \
load_dataset()

# example of a picture
index = 25
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index] + ", it's a '") + \
       classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' picture.")


### get sample sizes and dims of pictures
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
n_x = train_set_x_orig.shape[1]

### reshape the training and test examples
#A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b ∗∗ c ∗∗ d, a) is to use:
#X_flatten = X.reshape(X.shape[0], -1).T      # X.T is the transpose of X
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

## standardize the color value. the red, green and blue channels (RGB), ranging from 0 to 255
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

## define sigmoid
sigmoid = lambda z: 1. / (1. + np.exp(-z))
print("sigmoid([0, 2]) = " + str(sigmoid(np.array([0, 2])))) # must transform list to np array

## after the parameters are initialized, you can do the "forward" and "backward" propagation steps for learning the parameters.
## propagate
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation
    Argument:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector 0 if non-cat, 1 of cat
    
    Return:
    cost -- negative log likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w
    db -- gradient of the loss with respect to b
    """
    m = X.shape[1]
    
    A = sigmoid(np.dot(w.T, X) + b)
    
    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A -Y) / m
    
    
    cost = -(np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))) / m
    cost = np.squeeze(cost)
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    assert(cost.shape == ())
    
    grads = {"dw": dw, "db": db}
    
    return grads, cost


w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))


def optimize(w ,b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    Arguments:
    w -- weights, a numpy array
    b -- bias, a scalar
    X -- data
    Y -- label vector  0 if non-cat, 1 if cat
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Return:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of w and b with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve
    """
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}

    return params, grads, costs

params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))

## prediction

def predict(w, b, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    Arguments:
    w -- weights
    b -- bias
    X -- data

    Return:
    Y_prediction -- a numpy array containing all predictions
    """
    m = X.shape[1]
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) +b)

    Y_prediction = np.array([[int(A[0, i]>0.5) for i in range(A.shape[1])]])
    assert(Y_prediction.shape == (1, m))

    return Y_prediction

print ("predictions = " + str(predict(w, b, X)))

## Merge all functions into a model
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Argument:
    X_train -- training data
    Y_train -- labels for training data
    X_test -- test data
    Y_test -- labels for test data
    num_iterations --  hyperparameter, number of iterations
    learning_rate -- hyperparameter, learning rate
    print_cost -- set to true to print the cost every 100 iterations

    Return:
    d -- dictionary containing information about the model
    """
    w, b = np.random.randn(1, X_train.shape[0]) * 0.01, np.zeros((1, X_train.shape[0]))

    # gradient descent
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=False)

    # retrive parameters w and b from dictionary "parameters"
    w, b = params["w"], params["b"]

    # predict test/train examples
    Y_pred_test = predict(w, b, X_test)
    Y_pred_train = predict(w, b, X_train)

    # print train/test errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_pred_test,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)



