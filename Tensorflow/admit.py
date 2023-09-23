import tensorflow as tf
import pandas as pd

data = pd.read_csv('gpascore.csv')

data = data.dropna()

y데이터 = data['admit'].values
x데이터 = []

for i,rows in data.iterrows():
    x데이터.append([(rows['gre']),rows['gpa'],rows['rank']])

model = tf.keras.models.Sequential([ 
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
 ])

model.compile(optimizer='adam',loss='binary_crossmentropy',metrics=['accuracy'])

model.fit(x데이터,y데이터, epochs=1)

