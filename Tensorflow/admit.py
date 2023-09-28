import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv('gpascore.csv')

data = data.dropna()

y데이터 = data['admit'].values
x데이터 = [ ]

for i,rows in data.iterrows():
    x데이터.append([rows['gre'],rows['gpa'],rows['rank']])

model = tf.keras.models.Sequential([ 
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
 ])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(np.array(x데이터),np.array(y데이터), epochs=1000)

예측값 = model.predict( [ [750,3.20,3],[400,2.2,2]])

print(예측값)