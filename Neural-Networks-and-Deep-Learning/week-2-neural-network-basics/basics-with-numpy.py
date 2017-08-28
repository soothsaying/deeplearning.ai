## basic_sigmoid
import math
basic_sigmoid = lambda x: 1./(1. + math.exp(x))

## sigmoid from numpy array
import numpy as np
sigmoid = lambda x: 1./(1. + np.exp(x))
x = np.array([1, 2, 3])
sigmoid(x)

## sigmoid derivative
sigmoid_derivative = lambda x: sigmoid(x) * (1- sigmoid(x))
x = np.array([1, 2, 3])
print("sigmoid_derivative(x) = "+str(sigmoid_derivative(x)))

## reshaping arrays
## 3D array of shape(length,height,depth=3)
## convert it to a vector of shape (length∗height∗3,1)
image2vector = lambda image: image.reshape(image.shape[1]*image.shape[2]*image.shape[0], 1)
image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])
print("image2vector(x) = " + str(image2vector(image)))

## normalize Rows
def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x
    Argument: x - a numpy matrix of shape (n, m)
    Return: the normalized numpy matrix
    """
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x/x_norm
    return x
x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("normalizeRows(x) = " + str(normalizeRows(x)))

## softmax -- using broadcasting in numpy
def softmax(x):
    """
    Calculates the softmax for each row of the input x
    Argument: x -- a numpy matrix of shape (n, m)
    Return: A numpy matrix equal to the softmax of x, of shape (n, m)
    """
    x_exp = np.exp(x)
    x_sum_row = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum_row   ## (n, m) / (n ,1) = (n, m)
    return s

## vectorization
import time

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

## dot product using element-wise method
#tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot += x1[i] * x1[i]
#toc = time.process_time()
#print("dot = " + str(dot) + "takes " + str((toc-tic) * 1000) + "ms Computation time.")
print("dot = " + str(dot))

## dot product using vectorization
dot = np.dot(x1, x2)
print ("dot = " + str(dot))

## outer product using element-wise method
#tic = time.process_time()
outer = np.zeros((len(x1), len(x2)))
for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i, j] = x1[i] * x2[j]
print("outer = " + str(outer))

## outer product using vectorization
outer = np.outer(x1, x2)
print ("dot = " + str(outer))

## element-wise mutiplication using element-wise method
mul = np.zeros(len(x1))
for i in range(len(x1)):
    mul[i] = x1[i] * x2[i]
print("mul = " + str(mul))

## element-wise mutiplication using vectorization
mul = np.multiply(x1, x2)
print ("dot = " + str(mul))

## general dot using element-wise method
W = np.random.rand(3, len(x1)) #(3, length(x1))
gdot = np.zeros(W.shape[0])
for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i, j] * x1[j]
print("gdot = " + str(gdot))

## dot product using vectorization
W = np.random.rand(3, len(x1)) #(3, length(x1))
gdot = np.dot(W, x1)
print ("gdot = " + str(gdot))

## L1 norm
L1 = lambda yhat, y: np.sum(np.abs(yhat-y))

## L2 norm
L2 = lambda yhat, y: np.sum((yhat-y)**2)#np.sum(np.square(yhat-y))

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))
print("L2 = " + str(L2(yhat,y)))
