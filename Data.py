# Importing module

import os
import pandas as pd
import numpy as np

def getting_time_Series_Data(main_path = "C:\\Users\\velis\\Diploma_Thesis Detection\\data_ROM", num_sensor = 10, number_sol=100):

    len_of_ts = 1401
    matrix = np.zeros((num_sensor*(num_sensor-1)*number_sol,len_of_ts), dtype="float64")

    counter = 0
    for sol in range(number_sol): 
        for i in range(num_sensor): 
            files = os.listdir(os.path.join(main_path, ("sensor_"+str(i+1))))
            for j in files: 
                if j.startswith("actuator"):
                    frame = pd.read_csv(os.path.join(os.path.join(main_path, ("sensor_"+str(i+1))),j, "acceleration", "data.csv"), header=None)
                    matrix[counter,:] = frame.values[1:,sol].astype("float64") /np.max(frame.values[1:,sol].astype("float64"))
                    counter+=1
    return matrix

def train_test_split_(m, number_sensor=10, num_sol=100):

    healthy_solution = m[0:number_sensor*(number_sensor-1),:]
    damaged_solutions = m[number_sensor*(number_sensor-1):,:]

    train_damaged_sol = int((num_sol-1)*0.75)
    test_damaged_sol = (num_sol-1) - train_damaged_sol

    print(train_damaged_sol)
    print(test_damaged_sol)



    damaged_solution_train = damaged_solutions[0:train_damaged_sol*number_sensor*(number_sensor-1),:]
    damaged_solution_test = damaged_solutions[train_damaged_sol*number_sensor*(number_sensor-1):,:] 

    print("Len of healthy solutions", healthy_solution.shape)
    print("Len of damage solutions", damaged_solutions.shape)
    print("Len of damaged solutions train", damaged_solution_train.shape)
    print("Len of damaged solutions test", damaged_solution_test.shape)


    augmentation_factor = 5


    healthy_train_data = np.vstack(tuple([healthy_solution])*augmentation_factor*train_damaged_sol)
    damaged_train_data = np.vstack(tuple([damaged_solution_train])*augmentation_factor)
    healthy_test_data = np.vstack(tuple([healthy_solution])*augmentation_factor*test_damaged_sol)
    damaged_test_data = np.vstack(tuple([damaged_solution_test])*augmentation_factor)

    print(healthy_train_data.shape)
    print(damaged_train_data.shape)
    print(healthy_test_data.shape)
    print(damaged_test_data.shape)


    ## Adding Noise 

    noise_h_tr = []
    noise_d_tr = []
    noise_h_te = []
    noise_d_te = []

    for i in range(len(healthy_train_data)):
        noise_h_tr.append(np.random.normal(loc=0, scale=0.05, size=(healthy_train_data.shape[1],)))
        noise_d_tr.append(np.random.normal(loc=0, scale=0.05, size=(healthy_train_data.shape[1],)))
    
    for i in range(len(healthy_test_data)):
        noise_h_te.append(np.random.normal(loc=0, scale=0.05, size=(healthy_test_data.shape[1],)))
        noise_d_te.append(np.random.normal(loc=0, scale=0.05, size=(damaged_test_data.shape[1],)))
    
    noised_train_healthy = healthy_train_data+np.array(noise_h_tr)
    noised_train_damaged = damaged_train_data+np.array(noise_d_tr)
    noised_test_healthy = healthy_test_data+np.array(noise_h_te)
    noised_test_damaged = damaged_test_data+np.array(noise_d_te)

    noised_train_healthy = np.reshape(noised_train_healthy, newshape=(int(len(noised_train_healthy)/(number_sensor*(number_sensor-1))),(number_sensor*(number_sensor-1)), noised_train_healthy.shape[1]))
    noised_train_damaged = np.reshape(noised_train_damaged, newshape=(int(len(noised_train_damaged)/(number_sensor*(number_sensor-1))),(number_sensor*(number_sensor-1)), noised_train_damaged.shape[1]))
    noised_test_healthy = np.reshape(noised_test_healthy, newshape=(int(len(noised_test_healthy)/(number_sensor*(number_sensor-1))),(number_sensor*(number_sensor-1)), noised_test_healthy.shape[1]))
    noised_test_damaged = np.reshape(noised_test_damaged, newshape=(int(len(noised_test_damaged)/(number_sensor*(number_sensor-1))),(number_sensor*(number_sensor-1)),noised_test_damaged.shape[1]))

    return noised_train_healthy, noised_train_damaged, noised_test_healthy, noised_test_damaged