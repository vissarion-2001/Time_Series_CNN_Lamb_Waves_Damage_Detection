# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 17:41:23 2025

@author: velis
"""
import numpy as np 

import Time_Series_CNN_Lamb_Waves_Damage_Detection.TS_Detection as TS_Detection 
import Time_Series_CNN_Lamb_Waves_Damage_Detection.Data as Data 

if __name__ == "__main__" : 
    
    # train_h = np.load("f_final_h_train_aug.npy") 
    # train_d = np.load("f_final_d_train_aug.npy")
    # test_h = np.load("f_final_h_test_aug.npy")
    # test_d = np.load("f_final_d_test_aug.npy")

    matrix = Data.getting_time_Series_Data()
    ht, dt, he, de = Data.train_test_split_(matrix)
    
    detect_preprocessing_obj = TS_Detection.Loading_Preprocessing_data(ht, he, dt, de) 
    xtrain, xtest, ytrain, ytest = detect_preprocessing_obj.feature_and_labels_vectors()   
    
    train_test_data = (xtrain, xtest, ytrain, ytest)
    
    nn_detection_obj = TS_Detection.Convolutional_neural_network_architecture(lr=0.0001, epochs=20, batch_size=16, train_test_data=train_test_data) 
    cnn_model, cnn_hist = nn_detection_obj.cnn() 
    predictions = nn_detection_obj.predictions() 
    conf_matrix = nn_detection_obj.conf_matrix() 
    nn_detection_obj.plot_conf_matrix() 
    
    results_detection_obj = TS_Detection.Plotting_the_results(epochs=20, history=cnn_hist, predictions=predictions, true_labels=ytest) 
    results_detection_obj.plotting_loss() 
    results_detection_obj.plotting_precision() 
    results_detection_obj.plotting_accuracy() 
    results_detection_obj.plotting_recall()
    
    
    