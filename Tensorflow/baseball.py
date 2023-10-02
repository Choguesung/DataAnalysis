import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 데이터 로드
data = pd.read_csv('C:/github/DataAnalysis/Tensorflow/statiz_year_bat_0_1110.csv')  # 야구 선수 데이터 파일을 로드하세요

# 필요한 스탯 데이터 선택
selected_features = ['나이', 'G', '타석', '타수', '득점', '안타', '2타', '3타', '홈런', '루타', '타점', '도루', '도실', '볼넷', '사구', '고4', '삼진', '병살', '희타', '희비', '타율', '출루', '장타', 'OPS', 'wOBA', 'wRC+', 'WAR*', 'WPA']

# 입력 데이터와 타겟 데이터 분리
X = data[selected_features].values
y = data[selected_features].shift(-1).values  # 다음 년도의 스탯을 예측하므로 현재 스탯을 한 행씩 올립니다

# 결측치 처리
X = np.nan_to_num(X)  # 결측치를 0으로 대체하거나 다른 방식으로 처리하세요

# 데이터 스케일링 (정규화)
# 데이터 스케일링 (정규화)
scaler = MinMaxScaler()
X = scaler.fit_transform(X.reshape(-1, 1))  # 2D 배열로 변환하여 스케일링합니다


# 데이터 분할 (학습 및 테스트 세트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# LSTM 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(y_train.shape[1])  # 출력 레이어의 노드 수는 예측할 스탯 변수의 개수와 동일해야 합니다
])

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# LSTM 모델은 3D 입력을 요구하므로 데이터를 reshape합니다
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 모델 학습
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

