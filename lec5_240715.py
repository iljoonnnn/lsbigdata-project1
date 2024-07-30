import numpy as np

a = np.array([1,2])
b = np.array([1,2,3,4])
a+b

np.tile(a,2) +b

a = np.array([1.0, 2.0, 3.0])
b = 2.0
a.shape
b.shape

# 2차원 배열 생성
matrix = np.array([[ 0.0, 0.0, 0.0],
 [10.0, 10.0, 10.0],
 [20.0, 20.0, 20.0],
 [30.0, 30.0, 30.0]])
matrix.shape
# 1차원 배열 생성
vector = np.array([1.0, 2.0, 3.0, 4.0])
vector.shape
# 브로드캐스팅을 이용한 배열 덧셈
result = matrix + vector
print("브로드캐스팅 결과:\n", result)

# 세로 벡터 생성
vector = np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1)
vector
# 브로드캐스팅을 이용한 배열 덧셈
result = matrix + vector
print("브로드캐스팅 결과:\n", result)

### 여기까지가 복습
# 슬라이싱

import numpy as np
# 벡터 슬라이싱 예제, a를 랜덤하게 채움
np.random.seed(2024)
a = np.random.randint(1, 21, 10)
print(a)
# 두 번째 값 추출
print(a[1])

a[2:5]
a[-1] #맨 마지막에서 첫 번째 라는 뜻.
a[-2]
a[::2] #두 번씩 퐁당퐁당.
a[0:6:2]

#1에서부터 1000 사이에 3의 배수 합은?
sum(np.arange(0,1000,3))
x = np.arange(0,1001)
sum(x[::3])

print(a[[0, 2, 4]])

a
np.delete(a,[1,3])

a > 3
a[a>3] #사실 이거하려고 지금까지 위에거 배운거다.

np.random.seed(2024)
a = np.random.randint(1, 10000, 5)
type(a)

a[(a>2000)&(a<5000)]

import pydataset

df=pydataset.data('mtcars')
np_df=np.array(df['mpg'])

#15이상, 25이하인 데이터 갯수는?
sum((np_df>=15) & (np_df<=25))

# 평균 mpg 이상인인 애들은 몇대?
np_df_mean = sum(np_df) / len(np_df)
sum(np_df>=np_df_mean)

#15보다 작거나 22 이상인 데이터 갯수는?
sum((np_df<15) | (np_df>=22))

np.random.seed(2024)
a = np.random.randint(1, 10000, 5)
b = np.array(['a','b','c','f','w'])
a
b
b[(a>2000)&(a<5000)] #a의 원소들로 필터를 걸면서 b를 출력하는것!


model_names = np.array(df.index)
model_names[(np_df>=np_df_mean)]
model_names[(np_df<np_df_mean)]

a = np.array([1,2,3])
a.shape
a = a.reshape(1,3)
a.shape #reshape를 하면 2차원이된다!??!

df['mpg'][df['mpg']>30]

np.random.seed(2024)
a = np.random.randint(1,100,10)
a
np.where(a<50) #True인 인덱스(위치)를 반환해주라.

np.random.seed(2024)
a = np.random.randint(1,26346,1000)
a
#처음으로 22000보다 큰 숫자가 나오는 숫자는?
x = np.where(a>22000) #이게 튜플로 나옴 ㅠㅠㅠ
myindex = x[0][0] #튜플 안에 있는 어레이를 먼저 빼내고, 그 어레이의 0번 인덱스 값을 뽑아달라는 명령.
a[myindex]

#처음으로 10000보다 큰 숫자 중에서 50번째로로 나오는 숫자는?
np.random.seed(2024)
a = np.random.randint(1,26346,1000)
x = np.where(a>10000)[0]
x[49]
a[x[49]]

#500보다 작은 숫자들 중 가장 마지막으로 나오는 숫자
x = np.where(a<500)[0]
x[-1]

a = np.array([20, np.nan, 13, 24, 309])
a #타입 str아님?
a+3 #이게 더해지네 ㅋㅋㅋ

np.mean(a)
np.nanmean(a)
np.nan_to_num(a, nan = 0)

False
a = None
b = np.nan
b
a
b + 1
a

a = np.array([20, np.nan, 13, 24, 309])
np.isnan(a)

a_filtered = a[~np.isnan(a)]
a_filtered

str_vec = np.array(["사과", "배", "수박", "참외"])
str_vec
str_vec[[0, 2]]

mix_vec = np.array(["사과", 12, "수박", "참외"], dtype=str)
mix_vec

combined_vec = np.concatenate((str_vec, mix_vec))
combined_vec

col_stacked = np.column_stack((np.arange(1, 5), np.arange(12, 16)))
col_stacked

row_stacked = np.vstack((np.arange(1, 5), np.arange(12, 16)))
row_stacked

uneven_stacked = np.column_stack((np.arange(1, 5), np.arange(12, 18)))
uneven_stacked

vec1 = np.arange(1, 5)
vec2 = np.arange(12, 18)
vec1 = np.resize(vec1, len(vec2))
vec1
vec1 = np.resize(vec1, len(15))
vec1

import numpy as np
uneven_stacked = np.column_stack((vec1, vec2))
uneven_stacked

#주어진 백터에 5 더하기기
a = np.array([1, 2, 3, 4, 5])
a+5

#주어진 벡터의 홀수 번째 요소만 추출출
a = np.array([12, 21, 35, 48, 5])
#a[a % 2 == 1]
a[0::2]
a[1::2]

#주어진 백터에서 최대값.
a = np.array([1, 22, 93, 64, 54])
a.max()

#중복값 제거거
a = np.array([1, 2, 3, 2, 4, 5, 4, 6])
a
np.unique(a)

#번갈아서 새롭게 만들기기
a = np.array([21, 31, 58])
b = np.array([24, 44, 67])
x = np.arange(0, 6)
x[0::2] = a
x[1::2] = b
x

import numpy as np
import pandas as pd

df = pd.DataFrame({'name' : ['김지훈', '이유진', '박동현', '김민지'],
          'english' : [90, 80, 60, 70],
          'math' : [50, 60, 100, 20]})
df
type(df)
df["name"]
type(df["name"])
df["english"]
type(df["english"])
sum(df["english"])/4

df = pd.DataFrame({'제품': ['사과', '딸기', '수박'],
'가격': [1800,1500,3000],
'판매량': [24,38,13]})

sum(df['가격']) / 3
sum(df['판매량']) / 3

###판다스
import pandas as pd
df_exam=pd.read_excel("data/excel_exam.xlsx")

sum(df_exam["math"]) / 20
sum(df_exam["english"]) / 20
sum(df_exam["science"]) / 20

len(df_exam)
df_exam.shape
df_exam.size

df_exam['total'] = df_exam['math'] + df_exam['english'] + df_exam['science']
df_exam
df_exam['mean'] = df_exam['total'] / 3

df_exam[df_exam['math']>50]

df_exam[(df_exam['math']>50) & (df_exam["english"]>50)]

mean_m = np.mean(df_exam["math"])
mean_e = np.mean(df_exam["english"])

df_exam[(df_exam['math']>mean_m) & (df_exam["english"]<mean_e)]

class3 = df_exam[df_exam["nclass"] == 3]
class3[["math", "english", "science"]]
class3
class3["id"==10]

df_exam[0:10]
df_exam[7:16]

df_exam[1::2]

df_exam.sort_values("math")
df_exam.sort_values("math", ascending=False)
df_exam.sort_values(["nclass", "math"])

np.where(a>3, "Up", "Down")

df_exam["updown"] = np.where(df_exam["math"] >50, "Up", "Down")
df_exam
