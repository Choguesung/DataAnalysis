import pandas as pd
import tensorflow as tf

# 모델 불러오기
loaded_model = tf.keras.models.load_model('C:/github/DataAnalysis/Tensorflow/my_model')

# 테스트 데이터 불러오기
test_data = pd.read_csv('C:/github/DataAnalysis/Tensorflow/train.csv')

# 특성 전처리 (훈련 데이터와 동일한 전처리 단계를 적용)
평균 = test_data['Age'].mean()
test_data['Age'].fillna(value=평균, inplace=True)

# 나머지 특성 전처리 단계도 동일하게 적용 (Fare, Parch, SibSp, Sex, Embarked, Pclass, Ticket 등)

# 예측 수행
predictions = loaded_model.predict(test_data)

# 예측 결과 저장
test_data['Predicted_Survived'] = predictions
test_data.to_csv('result.csv', index=False)
