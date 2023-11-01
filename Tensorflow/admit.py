import tensorflow as tf
import pandas as pd
import numpy as np

# 데이터 불러오기
data = pd.read_csv('gpascore.csv')

# 누락된 데이터 제거
data = data.dropna()

# 입력 특성과 레이블 분리
y_data = data['admit'].values
x_data = [ ]

for i,rows in data.iterrows():
    x_data.append([ rows['gre'],rows['gpa'],rows['rank']])


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit( np.array(x_data), np.array(y_data), epochs=10000)

예측값 = model.predict([ [750, 3.70 ,3], [400, 2.2 ,1]])
print(예측값)




model.save('admit2.h5')

 