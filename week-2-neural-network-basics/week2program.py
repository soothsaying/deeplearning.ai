#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 17:36:45 2017

@author: wdsxsx
"""

test = "Hello World"
print ("test: "+test)

# graded function: basic_sigmoid

import math

def basic_sigmoid(x):
    """
    Compute sigmoid of x.
    
    Arguments:
    x -- A scalar
    
    Return:
    s -- sigmoid(x)
    
    """
    
    s=1/(1+math.exp(-x))
    
    return s

basic_sigmoid(3)

#x=[1,2,3]
#asic_sigmoid(x)    

import numpy as np

x = np.array([1, 2, 3])
print(np.exp(x))

x = np.array([1, 2, 3])
print(x+3)

def sigmoid(x):
    return 1/(1+np.exp(-x))

#sigmoid=lambda x: 1/(1+np.exp(-x))
sigmoid(x)

def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s * (1-s)
    return ds
# sigmoid_derivative = lambda x: sigmoid(x) * (1- sigmoid(x))

print ("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))

def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    v = image.reshape(image.shape[0]*image.shape[1]*image.shape[2], 1)
    
    return v

# image = 3*2*3
image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])
    
print ("image2vector(image) = " + str(image2vector(image)))

def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The  normalized (by Row) numpy matrix. You are allowed to modify x.
    
    """
    
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    
    x = x/x_norm #np broadcasting is used
    
    return x

x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("normalizeRows(x) = " + str(normalizeRows(x)))
        
def softmax(x):
    """
    Calculate the softmax for each row of the input x.
    
    Argument:
    x -- A numpuy matrix of shape (n, m)
    
    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n, m)
    """
    
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp/x_sum
    
    return s

x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(x)))
    

# Vectorization

import time

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

## dot product using loops
tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot += x1[i]*x2[i]
toc = time.process_time()
print ("dot = " + str(dot) + "\n ---- Computation time = " + str(1000*(toc-tic)) + "ms")

## outer product using loops
tic = time.process_time()
outer = np.zeros((len(x1), len(x2)))
for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i, j] = x1[i]*x2[j]
toc = time.process_time()
print ("outer = " + str(outer) + "\n ---- Computation time = " + str(1000*(toc-tic)) + "ms")

## elementwise implementation using loops
tic = time.process_time()
mul = np.zeros(len(x1))
for i in range(len(x1)):
    mul[i] = x1[i]*x2[i]
toc = time.process_time()
print ("elementwise implementation = " + str(mul) + "\n ---- Computation time = " + str(1000*(toc-tic)) + "ms")

## general dot using loops
W = np.random.randn(3, len(x1))
tic = time.process_time()
gdot = np.zeros(W.shape[0])
for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i, j]*x1[j]
toc = time.process_time()
print ("gdot = " + str(gdot) + "\n ---- Computation time = " + str(1000*(toc-tic)) + "ms")


# vectorized dot product 
tic = time.process_time()
dot = np.dot(x1, x2)
toc = time.process_time()
print ("dot = " + str(dot) + "\n ---- Computation time = " + str(1000*(toc-tic)) + "ms")

# vectorized outer product
tic = time.process_time()
outer = np.outer(x1, x2)
print ("outer= " + str(outer) + "\n ---- Computation time = " + str(1000*(toc-tic)) + "ms")

# vectorized elementwise multiplication
tic = time.process_time()
dot = np.multiply(x1, x2)
toc = time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ---- Computation time = " + str(1000*(toc-tic)) + "ms")

# vectorized general dot product
tic = time.process_time()
dot = np.dot(W, x1)
print ("gdot = " + str(gdot) + "\n ---- Computation time = " + str(1000*(toc-tic)) + "ms")

# L1 norm

L1 = lambda yhat, y: np.sum(np.absolute(y-yhat))
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))

# L2 norm

L2 = lambda yhat, y: np.sum(np.square(y-yhat))
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))



















