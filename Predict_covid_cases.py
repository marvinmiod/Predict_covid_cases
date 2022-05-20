# -*- coding: utf-8 -*-
"""
Created on Fri May 20 09:04:01 2022

This is a deep learning model using LSTM neural network to predict new cases 
(cases_new) in Malaysia using the past 30 days of number of cases.

# predict new cases 30 days only
# dont use batch normalization, never ever
# node set 64 and above
# dont add validation data in the model.fit
# sequence/time series dont do train/test split because its in sequence data
# training windows size 30 days, predict 1 day
#

@author: Marvin
"""

import re
import pandas as pd
import numpy as np
import datetime
import os
import pickle
import missingno as msno
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential #model is only for Sequential Model
from tensorflow.keras.layers import Dropout, Dense,  LSTM
from tensorflow.keras.layers import Embedding, Bidirectional
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Dropout # to forcefully explore diff route
from tensorflow.keras.layers import BatchNormalization # to add after hidden layer
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras import Input
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error

## import this in new file
#from Predict_covid_module import ModelCreation




#%%


# save the model and save the log
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'Predict_covid_cases.h5')
# path where the logfile for tensorboard call back
LOG_PATH = os.path.join(os.getcwd(),'Log_covid_cases')
log_files = os.path.join(LOG_PATH, 
                         datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))


DATASET_TRAIN_PATH = os.path.join(os.getcwd(),'dataset','cases_malaysia_train.csv')

DATASET_TEST_PATH = os.path.join(os.getcwd(),'dataset','cases_malaysia_test.csv')


#%% Def Function

def create_model(inputs):
        """
        This function creates the Sequential LSTM model
        and dropout layer

        Parameters
        ----------
        num_words : TYPE
            DESCRIPTION.
        num_categories : TYPE
            DESCRIPTION.
        embedding_output : TYPE, optional
            DESCRIPTION. The default is 128.
        nodes : TYPE, optional
            DESCRIPTION. The default is 64.
        dropout_value : TYPE, optional
            DESCRIPTION. The default is 0.2.

        Returns
        -------
        model : TYPE
            DESCRIPTION.

        """
        
        input_data_shape = x_train.shape[1] # this one is 30

        model = Sequential()
        #input node can be 64 or 128 or more
        model.add(LSTM(64, activation='tanh',
                       return_sequences=True, #return sequences is for 3 dimension data
                       input_shape=(input_data_shape,1))) # input shape of x train
        
        model.add(Dropout(0.2)) # dropout layer
        #model.add(LSTM(64, return_sequences=True)) # hidden layer
        model.add(LSTM(64))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='relu'))
        
        model.summary()
        plot_model(model)
        
        
        return model   
    

#%% EDA

# Step 1) Data Loading
df_train = pd.read_csv(DATASET_TRAIN_PATH)
df_test = pd.read_csv(DATASET_TEST_PATH)



#%%

# Step 2) Data inspection
df_train.info()
df_test.info()
df_train.describe().T
df_train['cases_new'].describe().T
df_train['cases_new'].dtypes
df_train.dtypes

# to create a backup copy
dummy_df_train = df_train.copy() 
dummy_df_test = df_test.copy()



#%%% Clean train data

# convert new cases in train data to numeric
df_train_case = pd.to_numeric(df_train['cases_new'], errors='coerce')


# inspect train data
df_train_case.isna().sum()
df_train_case.describe().T
df_train_case.info()
df_train_case.median() # 1331

plt.figure()
#plt.xlabel('date')
plt.plot(df_train_case)
plt.show


# clean test data
# convert new cases in test data to numeric
df_test_case = pd.to_numeric(df_test['cases_new'], errors='coerce')

# inspect train data
df_test_case.isna().sum()
df_test_case.describe().T
df_test_case.info()
df_test_case.median()

plt.figure()
#plt.xlabel('date')
plt.plot(df_test_case)
plt.show

#%% Fill NA with median value
# This code below is to fill in the missing value with median value 
# but it will skewed the graph as in low cases during early 2020  
# or high cases (20k) in 2021 will see a sudden drop to 1300 if use median
#df_date_new_cases.isna().sum()

#msno.bar(df_date_new_cases)
#msno.matrix(df_date_new_cases)


# there are some missing data in new cases column
# fill missing NAN value with mean
#new_cases_mean = dummy_df_train['cases_new'].median()

#df_date_new_cases = df_date_new_cases.fillna(dummy_df_train['cases_new'].mean())

# check back for NaN values and view graph for nan value
#df_date_new_cases.isna().sum()
#msno.matrix(df_date_new_cases)

#%% Drop NAN values for missing new cases
#med = df_train_case_drop['cases_new'].median()
# drop NaN values in both test and train data 
# as it contain 12 in train, 1 in test dataset
# it will skewed the graph as in low cases during early 2020 or high cases 
# or high cases 20k in 2021 will see it sudden drop to 1300
# Train data: drop NaN values from train data new_cases
df_train_case_drop = df_train_case.dropna()

# visualise the data after 12 data being drop
df_train_case_drop.isna().sum()

plt.figure()
#plt.xlabel('date')
plt.plot(df_train_case_drop)

#plt.plot(dummy_df_train['cases_new'])
plt.show

# TEST Data: drop NaN values from test data new_cases
df_test_case_drop = df_test_case.dropna()

# visualise the data after 1 data being drop from test dataset
df_test_case_drop.isna().sum()

plt.figure()
#plt.xlabel('date')
plt.plot(df_test_case_drop)
plt.show


#%% Step 6) Data Preprocessing

mms = MinMaxScaler()

x_train_scaled = mms.fit_transform(np.expand_dims(df_train_case_drop, -1))
x_test_scaled = mms.transform(np.expand_dims(df_test_case_drop, -1))

# create empty list first then append data into x train and y train
x_train = []
y_train = []

duration = 30 # this is to get the range of number 30 days

# to get the number of row from train data that has been cleaned(drop Na)
df_train_case_drop.shape[0]  # 668 in this case



# then append the data in the empty list for x and y train,
#for i in range(60,1258):
for i in range(duration, df_train_case_drop.shape[0]):
    x_train.append(x_train_scaled[i-duration:i,0])
    y_train.append(x_train_scaled[i,0])
    
# x_train_scaled[1198:1258]
# convert training and test [List] data to array
x_train = np.array(x_train) # this one is in 2 dimension
x_train = np.expand_dims(x_train, axis=-1) # expand to 3 Dimension

y_train = np.array(y_train)

#%% Prepare Testing Dataset

# x_train, x_test
# dataset_total = pd.concat((df_train,df_test),axis=0)
# combine both test and train data set that has been scaled ( 2 D shape)
duration = 30
dataset_total = np.concatenate((x_train_scaled, x_test_scaled), axis=0)

# to get length of duration plus x_test row of data
duration_plus_test = duration+len(x_test_scaled) # 129

# to get the last X dataset from the combined x train and x test data
last_dataset = dataset_total[-duration_plus_test:] 


# create empty list first then append data into x test and y test
x_test = []
y_test = []



#for i in range(duration,129):
for i in range(duration, duration_plus_test): # (30, 129) 129 is 30 + 99
    x_test.append(last_dataset[i-duration:i,0])
    y_test.append(last_dataset[i,0])


x_test = np.array(x_test)
x_test = np.expand_dims(x_test, axis=-1)

y_test = np.array(y_test)



#%% Model Creation

input_data_shape = x_train.shape[1] # this one is 30

model = Sequential()
#input node can be 64 or 128 or more
model.add(LSTM(64, activation='tanh',
               return_sequences=True, #return sequences is for 3 dimension data
               input_shape=(input_data_shape,1))) # input shape of x train

model.add(Dropout(0.2)) # dropout layer
#model.add(LSTM(64, return_sequences=True)) # hidden layer
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))

model.summary()
plot_model(model)


#%% Compile & Model training

# tensorboard callback
tensorboard_callback = TensorBoard(log_dir=log_files, histogram_freq=1)

# early stopping callback
early_stopping_callback = EarlyStopping(monitor='loss', patience=5 )

# Choose Mean Square Error (mse) because its not a regression problem
model.compile(optimizer='adam', loss='mse', metrics='mse') 

hist = model.fit(x_train, y_train, epochs=100, 
                 callbacks=[tensorboard_callback])

print(hist.history.keys())


#%% Visualise the model using matplotlib

# plot graph

plt.figure()
plt.plot(hist.history['loss'])
plt.title('Covid new cases vs training loss')
plt.xlabel('epoch')
plt.ylabel('training loss')
plt.show()


# to view the tensorboard go to browser type: http://localhost:6006/
# run in anaconda prompt tf_env: tensorboard --logdir "<path of log files>"

#%% Predicted value 

x_test_length = x_test.shape[0] # this is to get the size of x_test row (length)
# create empty list "predicted" and fit in the data in x_test
predicted = []
for i in x_test:
    predicted.append(model.predict(np.expand_dims(i, axis=0)))

# convert the predicted shape to 2D shape
predicted = np.array(predicted).reshape(x_test_length,1)

# plot the graph    
plt.figure()
plt.plot(predicted, color='r')
plt.plot(y_test, color='b')
plt.legend(['predicted', 'actual'])
plt.xlabel('Time')
plt.ylabel('Number of Covid19 Cases')
plt.show()    


#%%

# convert back minmax value back to original value for new cases
inversed_y_true = mms.inverse_transform(np.expand_dims(y_test, axis=-1))
inversed_y_predict = mms.inverse_transform(predicted)

#plot the graph with original  value

plt.figure()
plt.plot(inversed_y_predict, color='r')
plt.plot(inversed_y_true, color='b')
plt.legend(['predicted', 'actual'])
plt.xlabel('Time')
plt.ylabel('Number of Covid19 Cases')
plt.show()   

#%% Performance evaluation

y_true = y_test
#y_predicted = np.array(predicted).reshape(99,1)
y_predicted = predicted

print('\nMean Absolute Percentage Error (MAPE) is', mean_absolute_error(y_test, y_predicted)/sum(y_predicted) *100, '%')


