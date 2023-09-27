import numpy as np
import tensorflow as tf

pmodel = tf.keras.models.load_model('/Users/user/Documents/Github/DataAnalysis/Tensorflow/model1')

text = open('/Users/user/Documents/Github/DataAnalysis/Tensorflow/piano.txt','r').read()

unique_text = list(set(text))
unique_text.sort()

text_to_num = {}
num_to_text = {}

for i, data in enumerate(unique_text):
    text_to_num[data] = i
    num_to_text[i] = data

numeric_text = [text_to_num[i] for i in text]

first_input = numeric_text[117 : 117+25]
first_input = tf.one_hot(first_input, 31)
first_input = tf.expand_dims(first_input, axis=0)
# print(first_input)

music = []

for i in range(200):
   예측값 = pmodel.predict(first_input)
   예측값 = np.argmax(예측값[0])

   music.append(예측값)

   다음입력값 = first_input.numpy()[0][1:]

   one_hot_num = tf.one_hot(예측값, 31)

   first_input = np.vstack([다음입력값, one_hot_num.numpy()])
   first_input = tf.expand_dims(first_input, axis=0)

music_text = []

for i in music:
    music_text.append( num_to_text[i] )

print(''.join(music_text))