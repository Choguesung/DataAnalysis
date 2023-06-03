import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# csv 파일을 읽어옵니다.
df = pd.read_csv('baseball.csv')

# 입력 데이터와 타깃 데이터를 분리합니다.1
X = df[['teamA_war1', 'teamA_war2', 'teamA_war3', 'teamA_war4', 'teamA_war5', 'teamA_war6', 'teamA_war7', 'teamA_war8', 'teamA_war9', 'teamA_war10', 'teamB_war1', 'teamB_war2', 'teamB_war3', 'teamB_war4', 'teamB_war5', 'teamB_war6', 'teamB_war7', 'teamB_war8', 'teamB_war9', 'teamB_war10']]
y = df['teamA_win']

# 학습용 데이터와 검증용 데이터를 나눕니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 로지스틱 회귀 모델을 생성하고 학습합니다.
model = LogisticRegression()
model.fit(X_train, y_train)

# 검증용 데이터에 대해 예측을 수행합니다.
y_pred = model.predict(X_test)

# 정확도를 계산합니다.1
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
