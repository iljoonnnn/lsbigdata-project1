import pandas as pd
import numpy as np
#교제 6-7 데이터 합치기
test1 = pd.DataFrame({'id' : [1,2,3,4,5],
                    'midterm' : [60,80,70,90,85]})
test2 = pd.DataFrame({'id' : [1,2,3,4,5],
                     'final' : [70,83,65,95,80]})

total = pd.merge(test1,test2, how = "left", on = 'id')

test1 = pd.DataFrame({'id' : [1,2,3,4,5], 'midterm' : [60,80,70,90,85]})
test1
test2 = pd.DataFrame({'id' : [1,2,3,40,5],'final' : [70,83,65,95,80]})
test2

total = pd.merge(test1,test2, how = "left", on = 'id') #왼쪽의 아이디 기준준
total
total = pd.merge(test1,test2, how = "right", on = 'id') #오른쪽의 아이디 기준
total
total = pd.merge(test1,test2, how = "inner", on = 'id') #중복되는 아이디 기준준
total
total = pd.merge(test1,test2, how = "outer", on = 'id') #전체 아이디 기준준
total

exam = pd.read_csv('data/exam.csv')

name = pd.DataFrame({'nclass' : [1,2,3,4,5], 'teacher' : ['kim', 'lee','park','choi','jung']})
name

pd.merge(exam, name, how = 'left', on = 'nclass')

#세로로 쌓는 방법
score1 = pd.DataFrame({'id' : [1,2,3,4,5], 'score' : [60,80,70,90,85]})
score2 = pd.DataFrame({'id' : [6,7,8,9,10], 'score' : [70,83,65,95,80]})
score_all = pd.concat([score1, score2])

score_all2 = pd.concat([score1, score2], axis = 1) #그냥 옆으로 쌓고 싶을 때
score_all2

#데이터 정제
df = pd.DataFrame({
    'sex' : ['M', 'F', np.nan, 'M', 'f'],
    'score' : [5, 4, 3, 4, np.nan]
    })
df
df["score"] + 1
pd.isna(df).sum()

#결측치 제거하기
df.dropna() #모든 변수 결측치 제거거
df.dropna(subset = "score")
df.dropna(subset = ["score","sex"])

exam= pd.read_csv("data/exam.csv")

# 데이터 프레임 location을 사용한 인덱싱
#df.loc[행 인덱스, 열 컬럼 명명]
exam.loc[:3]
exam.iloc[:3]
b = exam.loc[[0], 'nclass']
b.info()
#숫자로만 하고 싶으면 df.iloc[]
a = exam.iloc[0:2, 0:4]
a.info()

#아무튼 교제로 다시 돌아와서서
exam.loc[[2,7,14], ["math"]] = np.nan

#문제 3을 필터링 하여 다른 숫자로 바꿔보기기
exam= pd.read_csv("data/exam.csv")
exam.loc[[2,7,14], ["math"]] = 3
a = exam.loc[[2,7,14], ["math"]]
exam

exam.loc[[exam["math"] == 3], "math"] = 1000

#수학점수 50점 이하인 학생들 점수를 50점으로 상향 조정!
exam.loc[exam['math'] <= 50, 'math'] = 50
exam

#영어점수 90점 이상을 90점으로 하향 조정 (iloc 사용)
exam.iloc[exam['english'] >= 90, 3] = 90
exam.iloc[exam['english'] >= 90, 3] #조회 안됨
exam.iloc[np.array(exam["english"] >= 90), 3] = 90 
exam.iloc[np.array(exam["english"] >= 90), 3] #조회 됨
exam

#math 점수 50점 이하는 "-"" 로 변경
exam.loc[exam['math'] <= 50, 'math'] = "-"
exam

#- 결측치를 수학점수 평균 바꾸고 싶은 경우
exam= pd.read_csv("data/exam.csv")
exam.loc[exam['math'] <= 50, 'math'] = "-"
exam

mean_math = exam.loc[exam['math'] != "-", "math"].mean() #아영 방법
mean_math = exam[exam['math'] != "-"]['math'].mean() #다른 방법법

exam.loc[exam['math'] == "-", 'math'] = mean_math #평균으로 교체하는 방법 1
exam["math"] = exam["math"].replace("-", mean_math) #평균으로 교체하는 방법 2
exam

