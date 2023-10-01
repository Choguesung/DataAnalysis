import tensorflow as tf

# 모델 로드하기
loaded_model = tf.keras.models.load_model('admit2.h5')

예측값 = loaded_model.predict([ [550, 3.70 ,3], [200, 4.2 ,3]])
print(예측값)