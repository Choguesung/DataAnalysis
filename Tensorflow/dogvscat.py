import os
import tensorflow as tf
import shutil

# print( len(os.listdir('C:/github/DataAnalysis/Tensorflow/dogs-vs-cats/train')))

# for i in os.listdir('C:/github/DataAnalysis/Tensorflow/dogs-vs-cats/train/'):
#     if 'cat' in i:
#         shutil.copyfile('C:/github/DataAnalysis/Tensorflow/dogs-vs-cats/train/'+ i,'C:/github/DataAnalysis/Tensorflow/dataset/cat/'+ i)
#     if 'dog' in i:
#         shutil.copyfile('C:/github/DataAnalysis/Tensorflow/dogs-vs-cats/train/'+ i,'C:/github/DataAnalysis/Tensorflow/dataset/dog/'+ i)

# 얘는 80퍼 먹음
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'C:/github/DataAnalysis/Tensorflow/dataset/',
    image_size=(64,64),
    batch_size=32,
    subset='training',
    validation_split=0.2, # 데이터를 20% 쪼개겠습니다
    seed=1234
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'C:/github/DataAnalysis/Tensorflow/dataset/',
    image_size=(64,64),
    batch_size=32,
    subset='validation',
    validation_split=0.2, # 데이터를 20% 쪼개겠습니다
    seed=1234
)

print(train_ds)