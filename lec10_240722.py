fruits = ["apple", "banana", "cherry"]
num = [1,2,3,4,5]
mixed = [1, "apple", 3.5, True]

# 빈 리스트 생성
empty_list1 = []
empty_list2 = list()

numbers = [1, 2, 3, 4, 5]
range_list = list(range(5))
range_list[3] = "LS 빅데이터 스쿨" #리스트 바꾸기기
range_list
range_list[1] = ["1st", "2nd", "3rd"] #리스트 원소를 리스트로 바꾸기.
range_list
range_list[1][2]

# 리스트 내포(comprehension)
# 1. 대괄호로 쌓여져있다 => 리스트다.
# 2. 넣고싶은 수식표현을 x를 사용해서 표현
# 3. for .. in .. 을 사용해서 원소정보 제공
# x를 제곱할건데 x는 여기서 가져올거야. 라는 뜻.
list(range(10))
squares = [x**2 for x in range(10)]
print("제곱 리스트:", squares)

# 3, 5, 2, 15의 세제곱을 하고 싶다면?
my_squares = [x**3 for x in [3, 5, 2, 15]]
my_squares
#결국 for x in이란 x를 옆에서 가져온다는 명령어구나!!

#넘파이도 될까?
import numpy as np
np.array([3, 5, 2, 15])
my_squares = [x**3 for x in np.array([3, 5, 2, 15])]
#출력값이 다르지만 가능하긴 하다.

#판다스 시리즈도 가능할까?
import pandas as pd
exam = pd.read_csv("data/exam.csv")
exam["math"] #컬럼명이 없어지고 밑에 dtype: 나오는건 판다스 시리즈
my_squares = [x**3 for x in exam["math"]]
#이것도 잘 먹구나.

3 + 2 #계산
"안녕" + "하세요" #문자열 붙여줌
["apple", "banana", "cherry"] + [1,2,3,4,5] #리스트도 붙여줌

"안녕" * 3 #문자열 반복
["apple", "banana", "cherry"] * 3 #반복하여 한 리스트로.

# 리스트 각 원소별 반복
numbers = [5, 2, 3]
repeated_list = [x for x in numbers for y in range(3)] # x의 값을 리스트에서 가져온다. 그 후 뒤에 리스트의 원소 수 만큼 반복한다.
repeated_list
repeated_list = [y for x in numbers for y in range(3)] # Y를 출력해달라니까 이렇게 됨.
repeated_list
#밑에서 다시 살펴볼거임.

# _(언더바)의 의미
# 앞에 나온 값을 가리킨다.
5+4
_ + 6 # _는 9를 의미

# 값 생략
a, _, b = (1, 2, 4) #그냥 변수로 만들어져버림 ;;

del _ #지정된 변수를 없애는 명령어.
_

### for 루프 문법
# for i in 범위: 
# 작동방식
for x in [4,1,2,3]:
    print(x)

for i in range(5):
    print(i**2)

#리스트를 하나 만들어서
#for 루프를 사용해서 2,4,6,8....20의 수를 채워넣으세요.

a = []
for i in range(2, 21, 2):
    a.append(i)
a

# 퀴즈: mylist_b의 홀수번 째 위치에 있는 숫자들만 mylist에 가져오기
# for문을 써서.
mylist_b = [2, 4, 6, 80, 10, 12, 24, 35, 23, 20]
mylist = [0] * 5

for i in range(5):
    mylist[i] = mylist_b[i*2]
mylist

### 리스트 컴프리헨션을 쓰는 방법.
# 바깥은 무조건 대괄호로 묶어줌: 리스트 반환하기 위해서
# 결과로 나타날 i를 먼저 쓴다. "i"
# for문을 쓴다."for i in"
# i가 가져올 리스트를 쓴다.
mylist=[i for i in [1,2,3,4,5]]

# 반복문 심화
for i in [0,1,2]:
    for j in [0,1]:
        print(i,j)

#위에서 봤던거 다시 이해하기
numbers = [5, 2, 3]
repeated_list = [x for x in numbers for j in range(3)]
repeated_list

for i in numbers:
    for j in range(3):
        print(i, j)

#만약에 i만 출력하게 한다면?
for i in numbers:
    for j in range(3):
        print(i)
#한 줄 반복은 리스트로 바로 만들 때만 유용한듯?


# 리스트 원소 포함 여부 확인
fruits = ["apple", "banana", "cherry"]
"banana" in fruits

[x == "banana" for x in fruits]

#한 줄을 여러줄로 바꾸면
mylist = []
for x in fruits:
    mylist.append(x == "banana")
mylist

#한결누나가 원하는건? 바나나의 위치를 알려달라.
fruits = ["apple", "apple", "banana", "cherry"]
"banana" in fruits
for x in fruits:
    if x == "banana":
        print()

#numpy 사용하여 바나나 위치 뱉어내기.
import numpy as np
fruits = ["apple", "apple", "banana", "cherry"]
# 1. 리스트를 어레이로
fruits = np.array(fruits)
int(np.where(fruits == "banana")[0][0])

#리스트 함수 중 리버스
fruits = ["apple", "apple", "banana", "cherry"]
fruits.reverse() #원소 거꾸로 해줌.
print(fruits)

fruits.append("pineapple") #원소추가
fruits

#굳이 앞에 추가하고 싶으면?
fruits.reverse()
fruits.append("grape")
fruits.reverse()
print(fruits)

# 원소 삽입
fruits.insert(2, "test")
print(fruits)

# 원소 제거
fruits.remove("test")
print(fruits)
fruits.remove("apple") #하나만 지워짐
print(fruits)

#넘파이로 지우기
import numpy as np
# 넘파이 배열 생성
fruits = np.array(["apple", "banana", "cherry", "apple", "pineapple"])
# 제거할 항목 리스트
items_to_remove = np.array(["banana", "apple"])
# 불리언 마스크 생성_논리형 백터 생성
mask = ~np.isin(fruits, items_to_remove)
# 불리언 마스크를 사용하여 항목 제거
filtered_fruits = fruits[mask]
print("remove() 후 배열:", filtered_fruits)

#교제 8장
import pandas as pd
mpg = pd.read_csv("data/mpg.csv")
mpg.shape
import seaborn as sns
import matplotlib.pyplot as plt

# 산점도
sns.scatterplot(data = mpg,
                x = "displ",
                y = "hwy",
                hue = "drv") \
    .set(xlim = [3,6], ylim = [10, 30])
plt.show()

# 막대 그래프
mpg["drv"].unique() #시리즈에서 원소의 종류 보기.
#drv: 자동차 구동 방식. f=전륜구동, r=후륜구동, 4=사륜구동

df_mpg = mpg.groupby("drv", as_index = False) \
    .agg(mean_hwy = ("hwy", "mean"))
plt.clf()
sns.barplot(data = df_mpg, x = "drv", y = "mean_hwy", hue = "drv")
plt.show()

plt.clf()
sns.barplot(data = df_mpg.sort_values("mean_hwy"),  #정렬 추가
            x = "drv", y = "mean_hwy", hue = "drv")
plt.show()

#빈도 막대 그래프. 208p.
df_mpg = mpg.groupby("drv", as_index = False) \
            .agg(n = ("drv", "count"))
df_mpg

plt.clf()
sns.barplot(data = df_mpg, x = 'drv', y = 'n')
plt.show()

plt.clf()
sns.barplot(data = df_mpg.sort_values("n"), x = 'drv', y = 'n')
plt.show()

#빈도 그래프
plt.clf()
sns.countplot(data = mpg, x = "drv")
plt.show()

# plotly를 사용한 산점도
import plotly.express as px
px.scatter(data_frame = mpg, x = 'cty', y = 'hwy', color = 'drv')
#plt.show 명령어 하는거 아님.

