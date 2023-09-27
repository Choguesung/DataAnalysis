import numpy as np
import tensorflow as tf

text = open('/Users/user/Documents/Github/DataAnalysis/Tensorflow/piano.txt','r').read()

unique_text = list(set(text))
unique_text.sort()

text_to_num = {}
num_to_text = {}

for i, data in enumerate(unique_text):
    text_to_num[data] = i
    num_to_text[i] = data

numeric_text = [text_to_num[i] for i in text]

X = []
Y = []

for i in range(0, len(numeric_text) - 25):
    X.append(numeric_text[i : i+25])
    Y.append(numeric_text[i + 25])

X = np.array(X)
Y = np.array(Y)

# 원-핫 인코딩
X = tf.one_hot(X, len(unique_text))
Y = tf.one_hot(Y, len(unique_text))

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(100, input_shape=(25, len(unique_text))),
    tf.keras.layers.Dense(len(unique_text), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, batch_size=64, epochs=30, verbose=2)
