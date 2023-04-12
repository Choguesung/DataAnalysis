from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 붓꽃 데이터셋을 로드합니다.
iris = load_iris()

# 입력 데이터와 타깃 데이터를 나눕니다.
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# 로지스틱 회귀 모델을 생성하고 학습합니다.
model = LogisticRegression()
model.fit(X_train, y_train)

# 테스트 데이터에 대해 예측을 수행합니다.
y_pred = model.predict(X_test)

# 정확도를 계산합니다.
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
