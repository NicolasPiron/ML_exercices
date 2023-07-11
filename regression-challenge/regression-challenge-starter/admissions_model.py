import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import InputLayer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score


def model_design(features):
    model = Sequential()
    model.add(InputLayer(input_shape=(features.shape[1])))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    opt=tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='mse', metrics=['mae', 'mse'], optimizer=opt)
    return model

dataset = pd.read_csv('/Users/nicolaspiron/Documents/Python_projects/DNN/regression-challenge/regression-challenge-starter/admissions_data.csv')
dataset = dataset.drop('Serial No.', axis=1)
print(dataset.head())

features = dataset.drop('Chance of Admit ', axis=1)
labels = dataset['Chance of Admit ']

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# scale the data
ct = ColumnTransformer([('standardize', StandardScaler(), ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research'])], remainder='passthrough')
features_train = ct.fit_transform(features_train)
features_test = ct.transform(features_test)

# fit the model
model = model_design(features_train)

# apply early stopping for efficiency
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

# fit the model with 100 epochs and a batch size of 8
# validation split at 0.25
history = model.fit(features_train, labels_train.to_numpy(), epochs=100, batch_size=8, verbose=1, validation_split=0.25, callbacks=[es])

# evaluate the model
val_mse, val_mae = model.evaluate(features_test, labels_test.to_numpy(), verbose = 0)

# view the MAE performance
print("MAE: ", val_mae)

# evauate r-squared score
y_pred = model.predict(features_test)

print(r2_score(labels_test,y_pred))



