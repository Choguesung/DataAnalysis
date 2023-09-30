import pandas as pd
import tensorflow as tf

# 데이터 불러오기
data = pd.read_csv('C:/github/DataAnalysis/Tensorflow/train.csv')

# 결측치 처리
평균 = data['Age'].mean()
data['Age'].fillna(value=평균, inplace=True)

최빈값 = data['Embarked'].mode()
data['Embarked'].fillna(value=최빈값[0], inplace=True)

# Survived 열을 정답으로 사용
정답 = data.pop('Survived')

# 데이터셋 생성
ds = tf.data.Dataset.from_tensor_slices((dict(data), 정답))

# Feature Columns 정의
feature_columns = []

feature_columns.append(tf.feature_column.numeric_column('Fare'))
feature_columns.append(tf.feature_column.numeric_column('Parch'))
feature_columns.append(tf.feature_column.numeric_column('SibSp'))

# Age를 버킷화하여 추가
Age = tf.feature_column.numeric_column('Age')
Age_bucket = tf.feature_column.bucketized_column(Age, boundaries=[10, 20, 30, 40, 50, 60])
feature_columns.append(Age_bucket)

# Sex를 카테고리화하여 추가
vocab = data['Sex'].unique()
cat = tf.feature_column.categorical_column_with_vocabulary_list('Sex', vocab)
one_hot = tf.feature_column.indicator_column(cat)
feature_columns.append(one_hot)

# Embarked를 카테고리화하여 추가
vocab = data['Embarked'].unique()
cat = tf.feature_column.categorical_column_with_vocabulary_list('Embarked', vocab)
one_hot = tf.feature_column.indicator_column(cat)
feature_columns.append(one_hot)

# Pclass를 카테고리화하여 추가
vocab = data['Pclass'].unique()
cat = tf.feature_column.categorical_column_with_vocabulary_list('Pclass', vocab)
one_hot = tf.feature_column.indicator_column(cat)
feature_columns.append(one_hot)

# Ticket을 임베딩하여 추가
vocab = data['Ticket'].unique()
cat = tf.feature_column.categorical_column_with_vocabulary_list('Ticket', vocab)
embedding = tf.feature_column.embedding_column(cat, dimension=9)  # 9차원 임베딩
feature_columns.append(embedding)

# 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(feature_columns),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 모델 컴파일 및 학습
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 데이터셋 배치 설정
ds_batch = ds.batch(32)

# 모델 학습
model.fit(ds_batch, shuffle=True, epochs=20)
