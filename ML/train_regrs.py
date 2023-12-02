from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import tensorflow as tf
from joblib import dump
import shared_vars as sv
from keras.regularizers import L1L2
from keras.losses import Huber
from keras.callbacks import EarlyStopping
from keras import optimizers

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('loss') < 0.06:  
            print("\nReached less than 0.02 loss so cancelling training!")
            self.model.stop_training = True

def train_LSTM_Regression(path: str):
    callbacks = MyCallback()
    # Load data from CSV file
    data = pd.read_csv(path, header=None)
    data = data.iloc[:, 180:] # delete first columns
    # Split data into features (X) and target variable (y)
    X = data.iloc[:, :-6]  # Select all columns except the last 6
    y = data.iloc[:, -6:]  # Select the last 6 columns as target variables

    # Normalize data
    sv.scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = sv.scaler.fit_transform(X)
    dump(sv.scaler, 'scaler.joblib')

    # Combine scaled features with target variables
    # scaled_data = np.concatenate([X_scaled, y], axis=1)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Reshape input to be 3D [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    opt = optimizers.Adam(learning_rate=0.0001)
    # Design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, kernel_regularizer=L1L2(l1=0.02, l2=0.02), activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(50,  return_sequences=False, kernel_regularizer=L1L2(l1=0.02, l2=0.02), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6))
    model.compile(loss=Huber(delta=1.8), optimizer=opt)

    # Fit network
    history = model.fit(X_train, y_train, epochs=700, batch_size=32, validation_data=(X_test, y_test), verbose=2, shuffle=True, callbacks=[callbacks])
    model.save('_models/my_model_3.keras')
    return model

def train_LSTM_Regression_2(path: str):

    callbacks = [MyCallback()]
    # Load data from CSV file
    data = pd.read_csv(path, header=None)
    data = data.iloc[:, 60:] # delete first columns
    # Split data into features (X) and target variable (y)
    X = data.iloc[:, :-6]  # Select all columns except the last 6
    y = data.iloc[:, -6:]  # Select the last 6 columns as target variables

    # Normalize data
    sv.scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = sv.scaler.fit_transform(X)
    dump(sv.scaler, 'scaler.joblib')

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Reshape input to be 3D [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    opt = optimizers.Adam(learning_rate=0.01)

    # Design network
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, kernel_regularizer=L1L2(l1=0.01, l2=0.01)))
    model.add(Dropout(0.3))
    model.add(LSTM(100, return_sequences=False, kernel_regularizer=L1L2(l1=0.01, l2=0.01)))
    model.add(Dropout(0.3))
    model.add(Dense(6))
    model.compile(loss=Huber(delta=0.5), optimizer=opt)

    # Fit network
    history = model.fit(X_train, y_train, epochs=5000, batch_size=64, validation_data=(X_test, y_test), verbose=2, shuffle=False, callbacks=callbacks)
    model.save('_models/my_model_3.keras')
    return model
