import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 탐색 함수수
#head()
#tail()
#shape
#ino()
#describe() 디스크라이브

exam = pd.read_csv('data/exam.csv')
exam.head()
exam.tail()
exam.shape
# 메서드 vs 속성(어트리뷰트)
# 메서드는 함수
# 속성은 변수

exam.info() #데이터 자체가 가진 정보보
exam.describe() # 데이터 요약

type(exam)
var = [1,2,3]
type(var)
exam.head()
var.head() # 에러

exam2 = exam.copy()
exam2 = exam2.rename(columns = {"nclass" : "class"})

exam2['total'] = exam['math'] + exam['english'] + exam['science']

exam2['test'] = np.where(exam2['total']>=200, '합격', '불합격')

plt.clf()
exam2["test"].value_counts()
exam2["test"].value_counts().plot.bar()
plt.show()

a = exam2["test"].value_counts()
?a.plot.bar
#검색할때는 이렇게 해야된다.

exam2["test2"] = np.where(exam2["total"] >= 200, 'A', np.where(exam2["total"] >= 100, 'B', 'C'))

exam2["test2"].isin(["A", "B"])

#한결누나 질문문
np.random.seed(2024)
a = np.random.randint(1, 21, 10)
print(a)
?np.random.randint
a = np.arange(100)
np.random.choice(a, 10, replace = False)


#챕터 6
#데이터 전처리 함수
# query()
# df[]
# sort_values()
# groupby()
# assign
# agg()
# merge()
# concat()

#쿼리, 조건에 맞는 행만만 가져오기.
exam.query('nclass == 1')
exam.query('nclass != 1')
exam.query('english <= 80')
exam.query("nclass not in [1,2]")
exam[~exam['nclass'].isin([1,2])]

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


#필요한 변수(컬럼)만 가져오기.
exam[["id", "nclass"]]
exam[["nclass"]]
exam.drop(columns = "math")
exam

exam.query("nclass == 1")[["math", "english"]].sort_values("math")
exam.query("nclass == 1").sort_values("math").iloc[:,2:4]


# 오름차순 or 내림차순으로 정렬하기
exam.sort_values("math")
exam.sort_values("math", ascending = False)
exam.sort_values(["nclass", "english"], ascending = False)

#파생변수
#강사님 방식은 exam["total"] =
#여기 교제에서는
exam.assign(total = exam["math"] + exam["english"] + exam["science"])
exam = exam.assign(total = exam["math"] + exam["english"] + exam["science"])

exam2 = exam.copy()
exam2.groupby('nclass').agg(mean_math = ("math", "mean"))

exam.groupby('nclass').agg(mean_total = ("total", "mean"))

#166p 혼자서 해보기
#1, 2
import pydataset
mpg = pydataset.data("mpg")
mpg = mpg.rename(columns = {'class' : 'category'})
mpg.agg(mean_cty = ('cty', 'mean'))
mpg.groupby('category').agg(mean_cty = ('cty', 'mean'))
mpg.groupby('category').agg(mean_cty = ('cty', 'mean')).sort_values('mean_cty', ascending = False)

#3
mpg.groupby('manufacturer').agg(mean_hwy = ('hwy', 'mean')).sort_values('mean_hwy', ascending = False).head(3)

#4
mpg.query('category == "compact"').groupby('manufacturer').agg(count_compact = ('category', 'count')).sort_values('count_compact', ascending = False)

#연습
a = mpg.query('category == "compact"').groupby('manufacturer').agg(count_compact = ('category', 'count')).sort_values('count_compact', ascending = False)
a.transpose() #행렬전환

