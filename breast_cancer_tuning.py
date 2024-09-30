#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:50:40 2023

@author: josealvesdacunha
"""

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def createNeuralNeuralNetwork(optimizer, loss, kernel_initializer, activation, neurons):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, 
                          activation = activation, 
                          kernel_initializer = kernel_initializer, 
                          input_dim = 30))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, 
                          activation = activation, 
                          kernel_initializer = kernel_initializer))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 1, activation = 'sigmoid'))

    classificador.compile(optimizer = optimizer, 
                          loss = loss, 
                          metrics = ['binary_accuracy'])
    return classificador

 classificador = KerasClassifier(build_fn = createNeuralNeuralNetwork)
 params = {'batch_size': [10, 30],
               'epochs': [50, 100],
               'optimizer': ['adam', 'sgd'],
               'loss': ['binary_crossentropy', 'hinge'],
               'kernel_initializer': ['random_uniform', 'normal'],
               'activation': ['relu', 'tanh'],
               'neurons': [16, 8]}
 gridSearch = GridSearchCV(estimator = classificador, param_grid = params, scoring = 'accuracy', cv = 5)
 
 gridSearch = gridSearch.fit(previsores, classe)
 best_params = gridSearch.best_params_
 best_precision = gridSearch.best_score_