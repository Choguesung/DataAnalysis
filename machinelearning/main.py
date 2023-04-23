import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# 데이터 로드
data = pd.read_csv("war_data.csv")

# 입력 데이터와 타겟 데이터로 분리
X = data[["team1_war", "team2_war"]]
y = data["team1_win_ratio"]

# 데이터 분할 왜?
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 정의
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(2,)),
    keras.layers.Dense(1)
])

# 모델 컴파일
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# 모델 훈련
model.fit(X_train, y_train, epochs=100)
