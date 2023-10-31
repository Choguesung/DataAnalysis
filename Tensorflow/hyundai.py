import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import lightgbm as lgb
import bisect
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

# 데이터 로드
train = pd.read_csv('/content/sample_data/train.csv').drop(columns=['SAMPLE_ID'])  # train_data.csv 파일에 실제 데이터가 저장되어야 합니다.
test = pd.read_csv('/content/sample_data/test.csv').drop(columns=['SAMPLE_ID'])    # test_data.csv 파일에 테스트 데이터가 저장되어야 합니다.

print(train.columns)
print(test.columns)

# datetime 컬럼 처리
train['ATA'] = pd.to_datetime(train['ATA'])
test['ATA'] = pd.to_datetime(test['ATA'])

# datetime을 여러 파생 변수로 변환
for df in [train, test]:
    df['year'] = df['ATA'].dt.year
    df['month'] = df['ATA'].dt.month
    df['day'] = df['ATA'].dt.day
    df['hour'] = df['ATA'].dt.hour
    df['minute'] = df['ATA'].dt.minute
    df['weekday'] = df['ATA'].dt.weekday

# datetime 컬럼 제거
train.drop(columns='ATA', inplace=True)
test.drop(columns='ATA', inplace=True)

# Categorical 컬럼 인코딩
categorical_features = ['ARI_CO', 'ARI_PO', 'SHIP_TYPE_CATEGORY', 'ID', 'SHIPMANAGER', 'FLAG']
encoders = {}

for feature in tqdm(categorical_features, desc="Encoding features"):
    le = LabelEncoder()
    train[feature] = le.fit_transform(train[feature].astype(str))
    le_classes_set = set(le.classes_)
    test[feature] = test[feature].map(lambda s: '-1' if s not in le_classes_set else s)
    le_classes = le.classes_.tolist()
    bisect.insort_left(le_classes, '-1')
    le.classes_ = np.array(le_classes)
    test[feature] = le.transform(test[feature].astype(str))
    encoders[feature] = le

# 결측치 처리
train.fillna(train.mean(), inplace=True)
test.fillna(train.mean(), inplace=True)

def train_and_evaluate(model, model_name, X_train, y_train):
    print(f'Model Tune for {model_name}.')
    model.fit(X_train, y_train)

    feature_importances = model.feature_importances_
    sorted_idx = feature_importances.argsort()

    plt.figure(figsize=(10, len(X_train.columns)))
    plt.title(f"Feature Importances ({model_name})")
    plt.barh(range(X_train.shape[1]), feature_importances[sorted_idx], align='center')
    plt.yticks(range(X_train.shape[1]), X_train.columns[sorted_idx])
    plt.xlabel('Importance')
    plt.show()

    return model, feature_importances

X_train = train.drop(columns='CI_HOUR')
y_train = train['CI_HOUR']

# Model Tune for LGBM
lgbm_model, lgbm_feature_importances = train_and_evaluate(lgb.LGBMRegressor(), 'LGBM', X_train, y_train)
feature_importances = lgbm_model.feature_importances_

# 중요도를 특성과 함께 출력
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
sorted_feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(sorted_feature_importance_df)

threshold = 100 # Your Threshold
low_importance_features = X_train.columns[lgbm_feature_importances < threshold]

X_train_reduced = X_train.drop(columns=low_importance_features)

X_test_reduced = test.drop(columns=low_importance_features)

lgbm = lgb.LGBMRegressor(random_state=42, max_depth=-1, num_leaves=25, learning_rate=0.7,n_estimators=1000,boosting_type="dart", min_child_samples=56)

# 5-Fold 교차 검증
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

# 각 fold에 대한 예측 수행
ensemble_predictions = np.zeros(X_test_reduced.shape[0])

for train_idx, val_idx in kf.split(X_train_reduced):
    X_t, X_val = X_train_reduced.iloc[train_idx], X_train_reduced.iloc[val_idx]
    y_t, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # LightGBM 모델 학습
    lgbm.fit(X_t, y_t)
    
    # Validation set에 대한 예측
    val_pred = lgbm.predict(X_val)
    
    scores.append(mean_absolute_error(y_val, val_pred))

    # 테스트 데이터에 대한 예측
    test_pred = lgbm.predict(X_test_reduced)
    ensemble_predictions += test_pred

# K-fold 예측의 평균 계산
ensemble_predictions /= 5

# 각 fold에서의 Validation Metric Score와 전체 평균 Validation Metric Score 출력
print("Validation : MAE scores for each fold:", scores)
print("Validation : MAE:", np.mean(scores))