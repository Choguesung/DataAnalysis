import pandas as pd

df = pd.read_csv('score.csv')

print(df)

#DataFrame 확인
#계산가능한 데이터에 대해 갯수, 평균, 표준편차, 최소/최대값 등의 정보를 보여줌11
print(df.describe())

print(df.info)

#처음 5개의 row를 가져옴
print(df.head())

#꼬리 5개 (마지막)
print(df.tail())

#어떤 값들을 가지고 있는지
print(df.values)

#인덱스 정보
print(df.index)

#어떤 열들이 있는지
print(df.columns)

#크기 확인 (행 ,열 출력)
print(df.shape)

#Series 확인
print(df['키'].describe())

#키큰 순서대로 n명 확인
print(df['키'].nlargest(3))

#최대 최소
print(df['키'].mean())

#개수 확인
print(df['SW특기'].count())

#유일한거
print(df['학교'].unique())

#학교가 몇개인가? 중복제외하고
print(df['학교'].nunique())
