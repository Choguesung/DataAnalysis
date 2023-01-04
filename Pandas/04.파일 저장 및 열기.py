# DataFrame 객체를 엑셀, csv, txt 형태의 파일로 저장 및 열기
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

df = pd.DataFrame(data,index=['1번','2번','3번','4번','5번','6번','7번','8번'])
df.index.name='지원번호'

## 저장하기
## csv 파일로 저장
df.to_csv('score.csv')

# 인덱스 없이도 가능
df.to_csv('score.csv',index=False)

#텍스트 (.txt) 파일로 저장
df.to_csv('score.txt',sep='\t')# tab으로 구분된 텍스트 파일?

## 엑셀 파일로 저장
##df.to_excel('score.xlsx')

## csv파일 열기
df=pd.read_csv('score.csv')
print(df)

## 행 스킵
df=pd.read_csv('score.csv',skiprows=1) # 1개 행 무시 (지정된 개수 만큼의 row를 건너 뛴대요)
print(df)

df=pd.read_csv('score.csv',skiprows=[1,3,5]) # 원하는 행 무시
print(df)

df=pd.read_csv('score.csv',nrows=4) # 지정된 개수 만큼의 row만 가져옴
print(df)

