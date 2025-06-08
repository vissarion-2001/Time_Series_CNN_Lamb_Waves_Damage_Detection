# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 23:34:47 2024

@author: velis
"""

# Importing Libraries

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Importing Tools

from sklearn.preprocessing import MinMaxScaler
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Reshape, LSTM, Input, AveragePooling2D
from keras.models import Sequential, Model
from sklearn.metrics import confusion_matrix

class Loading_Preprocessing_data:
    
    def __init__(self, train_healthy_data, test_healthy_data, train_damage_data, test_damage_data):
        self.train_healthy_data = train_healthy_data
        self.train_damage_data = train_damage_data
        self.test_healthy_data = test_healthy_data
        self.test_damage_data = test_damage_data 
        self.__number_of_solutions_train = len(self.train_healthy_data)
        self.__number_of_solutions_test = len(self.test_healthy_data)
    
    def feature_and_labels_vectors(self):
        
        X_healthy_train = self.train_healthy_data[:, :, 0:-1:5] 
        X_damaged_train = self.train_damage_data[:, :, 0:-1:5] 
        X_healthy_test = self.test_healthy_data[:, :, 0:-1:5]
        X_damaged_test = self.test_damage_data[:, :, 0:-1:5] 
        
        print(X_healthy_train.shape) 
        print(X_damaged_train.shape) 
        print(X_healthy_test.shape) 
        print(X_damaged_test.shape)
        
        
        total_features_training = np.vstack((X_healthy_train, X_damaged_train))
        total_features_test = np.vstack((X_healthy_test, X_damaged_test)) 
        
        print(total_features_training.shape) 
        print(total_features_test.shape)
        
        # Reshaping for Convolutional Neural Network 
        total_features_r_train = np.reshape(total_features_training, newshape=(2*self.__number_of_solutions_train, 90, 280, 1))
        total_features_r_test = np.reshape(total_features_test, newshape=(2*self.__number_of_solutions_test, 90, 280, 1))
        
        # Labels 
        Labels_healthy_train = np.zeros((self.__number_of_solutions_train, 1)) 
        Labels_damaged_train = np.ones((self.__number_of_solutions_train, 1)) 
        Labels_healthy_test = np.zeros((self.__number_of_solutions_test, 1)) 
        Labels_damaged_test = np.ones((self.__number_of_solutions_test, 1)) 
        
        total_labels_train = np.vstack((Labels_healthy_train, Labels_damaged_train)).astype("int") 
        total_labels_test = np.vstack((Labels_healthy_test, Labels_damaged_test)).astype("int")
        
        return total_features_r_train, total_features_r_test, total_labels_train, total_labels_test
    
class Convolutional_neural_network_architecture:
    
    def __init__(self, lr, epochs, batch_size, train_test_data):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_test_data = train_test_data
    
    def cnn(self): 
        
        inputs = Input((90, 280, 1)) 
        x = Conv2D(filters=32, kernel_size=(3, 3), padding="SAME", activation="relu",)(inputs) 
        x = Conv2D(filters=32, kernel_size=(3, 3), padding="SAME", activation="relu")(x) 
        x = MaxPooling2D((2, 2))(x) 
        x = Flatten()(x) 
        x = tf.keras.layers.BatchNormalization()(x) 
        x = Dense(units=128, activation="swish")(x) 
        x = Dropout(0.2)(x) 
        x = Dense(units=64, activation="swish")(x) 
        x = Dropout(0.2)(x) 
        x = Dense(units=32, activation="swish")(x) 
        x = Dropout(0.2)(x) 
        x = Dense(units=16, activation="swish")(x) 
        x = Dropout(0.2)(x) 
        Logits = Dense(units=1, activation="sigmoid")(x)
        
        CNN = Model(inputs, Logits)
        
        CNN.compile(optimizer=tf.keras.optimizers.Adam(self.lr), loss="binary_crossentropy", metrics=["Precision", "accuracy", "Recall"])
        
        history = CNN.fit(self.train_test_data[0], self.train_test_data[2], epochs=self.epochs, batch_size=self.batch_size, validation_data=(self.train_test_data[1], self.train_test_data[3]))
        
        return CNN, history
    
    def predictions(self):
        testing_input_data = self.train_test_data[1]
        predictions = self.cnn()[0].predict(testing_input_data)
        predictions = (predictions>0.5).astype("int")
        return predictions
    
    def conf_matrix(self):
        cm = confusion_matrix(self.train_test_data[3], self.predictions())
        return cm
        
    def plot_conf_matrix(self):
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.conf_matrix(), annot=True, fmt="d", cmap="viridis", xticklabels=["Healthy", "Damaged"], yticklabels=["Healthy", "Damaged"])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix Heatmap')
        plt.show()
    
class Plotting_the_results:
    
    def __init__(self, epochs, history, predictions, true_labels):
        self.epochs = epochs
        self.history = history
        self.predictions = predictions
        self.true_labels = true_labels
        
    def plotting_loss(self):
        a = np.arange(1, self.epochs+1)
        plt.plot(a, self.history.history['loss'])
        plt.plot(a, self.history.history['val_loss'])
        plt.legend(['loss', 'val_loss'])
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.title("CNN - Time Series Data - Damage Detection - D1 Dataset - Loss")
        plt.show()
    
    def plotting_precision(self):
        a = np.arange(1, self.epochs+1) 
        plt.plot(a, self.history.history['Precision']) 
        plt.plot(a, self.history.history['val_Precision']) 
        plt.legend(['precision', 'val_precision']) 
        plt.ylabel('precision') 
        plt.xlabel('Epochs') 
        plt.show()
        
    def plotting_accuracy(self): 
        a = np.arange(1, self.epochs+1) 
        plt.plot(a, self.history.history['accuracy']) 
        plt.plot(a, self.history.history['val_accuracy']) 
        plt.legend(['accuracy', 'val_accuracy']) 
        plt.ylabel('accuracy') 
        plt.xlabel('Epochs') 
        plt.show()
        
    def plotting_recall(self):
        a = np.arange(1, self.epochs+1) 
        plt.plot(a, self.history.history['Recall']) 
        plt.plot(a, self.history.history['val_Recall']) 
        plt.legend(['recall', 'val_recall']) 
        plt.ylabel('recall') 
        plt.xlabel('Epochs') 
        plt.show()
        
        
            
        
    
    
        
        
    
    