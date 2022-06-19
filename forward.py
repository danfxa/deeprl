#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 23:50:31 2022

@author: danang
"""
import numpy as np
from activation import *

def init_params(X):
    parameters = {}
    L = len(layer_dims)
    n_x = len(X)
    parameters['W' + str(0)] = np.random.rand(layer_dims[0], n_x)*0.01
    parameters['b'+ str(0)] = np.zeros((layer_dims[0], 1))

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.rand(layer_dims[l], layer_dims[l-1])*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def lin_forward(A,W,b):
    Z = np.dot(W,A)+ b
    cache = (A,W,b)
    return Z, cache
def lin_act_forward(A_prev,W,b,act):
    if act == 'sigmoid':
        Z, lin_cache = lin_forward(A_prev, W, b)
        A, act_cache = sigmoid(Z)
    elif act == 'relu':
        Z, lin_cache = lin_forward(A_prev, W, b)
        A, act_cache = relu(Z)
    cache = (lin_cache, act_cache)
    return A, cache

def forward(X, layer_dims):
    A = X    
    parameters = {}
    L = len(layer_dims)
    n_x = len(A)
    caches = []
    parameters['W' + str(0)] = np.random.rand(layer_dims[0], n_x)*0.01
    parameters['b'+ str(0)] = np.zeros((layer_dims[0], 1))

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.rand(layer_dims[l], layer_dims[l-1])*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    # step forward
    
    l = (len(parameters) //2)-1
    
    for i in range(l):
        A_prev = A
        A,cache = relu(np.dot(parameters['W' + str(i)], A_prev) + parameters['b' + str(i)])
        caches.append(cache)
    AL, cache = sigmoid(np.dot(parameters['W' + str(l)] , A) + parameters['b' + str(l)])
    caches.append(cache)
    return AL