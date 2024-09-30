#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:12:41 2023

@author: josealvesdacunha
"""

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def createNeuralNeuralNetwork():
    classificador = Sequential()
    classificador.add(Dense(units = 16, 
                          activation = 'relu', 
                          kernel_initializer = 'random_uniform', 
                          input_dim = 30))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 16, 
                          activation = 'relu', 
                          kernel_initializer = 'random_uniform'))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    otimizador = keras.src.optimizers.Adam(learning_rate = 0.001, decay = 0.0001, clipvalue = 0.5)
    classificador.compile(optimizer = otimizador, 
                          loss = 'binary_crossentropy', 
                          metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = createNeuralNeuralNetwork, epochs = 100, batch_size = 10)

resultado = cross_val_score(estimator = classificador, X = previsores, y = classe, cv = 10, scoring= 'accuracy')

media = resultado.mean()

desvio = resultado.std()