import pandas as pd

data = {
    '이름' : ['채치수', '정대만', '송태섭', '서태웅', '강백호', '변덕규', '황태산', '윤대협'],
    '학교' : ['북산고', '북산고', '북산고', '북산고', '북산고', '능남고', '능남고', '능남고'],
    '키' : [197, 184, 168, 187, 188, 202, 188, 190],
    '국어' : [90, 40, 80, 40, 15, 80, 55, 100],
    '영어' : [85, 35, 75, 60, 20, 100, 65, 85],
    '수학' : [100, 50, 70, 70, 10, 95, 45, 90],
    '과학' : [95, 55, 80, 75, 35, 85, 40, 95],
    '사회' : [85, 25, 75, 80, 10, 80, 35, 95],
    'SW특기' : ['Python', 'Java', 'Javascript', '', '', 'C', 'PYTHON', 'C#']
}

#print(data['이름'])

df = pd.DataFrame(data)

print(df)

# 데이터 접근1
print(df['이름'])
print(df['키'])

# 두개 이상 가져올때는 대괄호 두개로 감싸기1
print(df[['이름','키']])

#DataFrame 객체 생성 (Index 지정)

df = pd.DataFrame(data,index=['1번','2번','3번','4번','5번','6번','7번','8번'])
print(df)

#dataFrame 객체 생성 (Column 지정)
#이름,키만 필요해 등등..

df = pd.DataFrame(data,columns=['이름','학교x`'])
print(df)


