a=(1,2,3)

a=1,2,3 #이것도 튜플로 만들어진다.

def min_max(numbers):
  return min(numbers), max(numbers)

a = [1,2,3,4,5]
min_max(a)
#이렇게 하면 결과가 튜플로 나옴.
print(min_max(a))
#위에서 함수 설정 할 때 튜플로 묶지 않았음. 그래서 1,5 이렇게 결과가 나왔을 것임. 기본이 튜플이기 때문에 튜플로 만들어진것임.
######################################

a = [1,2,3]
b = a

a[1] = 4
a
b
#b도 같이 바뀌는건 아래 개념 때문에.
# soft copy, deep copy
id(a)
id(b)
id(a) == id(b)

#우리가 변수를 만들면, 어떤 공간안에 변수 안에 리스트를 입력한다. 그리고 그 공간의 위치(주소)를 a와 연동시킨다.
#그러면 a는 그 위치 값을 가지고 있는 것이다.
#b=a 라고 하면, b도 그 주소값을 가지게 되는 것이다.
#"a의 어떤 값을 바꿔줘." 라고하면, a가 가진 주소와 맞는 공간으로 가서 거기 있는 요소를 바꾸는 것이다.

#그러면 b에서 바꿔도 a가 바뀌겠네?
a = [1,2,3]
b=a
a
b
b[1] = 4
#실제로 바뀐다 ㅋㅋㅋ

#deep copy
a = [1,2,3]
a

b=a[:] #첫 번째 방법. 콜론 앞뒤로 아무것도 입력 안 하면 그대로 가져옴.

id(a)
id(b)
id(a) == id(b)

a[1] = 4
a
b

b=a.copy() #두 번째 방법. 복사라는 명령어 사용하기.
a
b
a == b
id(a) == id(b)

# 수학함수
x = 4
math.sqrt(x)
import math

# 제곱근 계산
sqrt_val = math.sqrt(16)
print("16의 제곱근은:", sqrt_val)

# 지수 계산
exp_val = math.exp(5)
print("e^5의 값은:", exp_val)

# 로그 계산
log_val = math.log(10, 10)
print("10의 밑 10 로그 값은:", log_val)

# 팩토리얼 계산
fact_val = math.factorial(5)
print("5의 팩토리얼은:", fact_val)

# 사인 함수 계산
sin_val = math.sin(math.radians(90)) # 90도를 라디안으로 변환
print("90도의 사인 함수 값은:", sin_val)

# 코사인 함수 계산
cos_val = math.cos(math.radians(180))
print("180도의 코사인 함수 값은:", cos_val)
# 탄젠트 함수 계산

tan_val = math.tan(math.radians(45))
print("45도의 탄젠트 함수 값은:", tan_val)

x = 1
mu = 0
sigma = 1

part_1 = 1/ (sigma * math.sqrt(2*math.pi))
part_2 = math.exp(((x-mu)**2) / (sigma**2) / (-2))
result = part_1 * part_2


x=2
y=9
z= math.pi / 2

(x**2 + math.sqrt(y) + math.sin(z)) * math.exp(x)

def my_g(x):
  return math.cos(x) + math.sin(x) * math.exp(x)

my_g(math.pi)

def fname(`indent('.') ? 'self' : ''`):
    """docstring for fname"""
    # TODO: write code...

# import pandas as pd
# 
# import numpy as np
# import numpy as np
# inport seaborn as sns

# import seaborn as sns

# Ctrl + Shift + C : 커멘트 처리
import numpy as np

# 벡터 생성하기 예제
a = np.array([1, 2, 3, 4, 5]) # 숫자형 벡터 생성
b = np.array(["apple", "banana", "orange"]) # 문자형 벡터 생성
c = np.array([True, False, True, True]) # 논리형 벡터 생성
print("Numeric Vector:", a)
print("String Vector:", b)
print("Boolean Vector:", c)
a
b
c
type(a)
#넘파이로 만든 어레이는 다른 타입임.
#이렇게 변경해야 넘파이 함수를 쓸 수 있다.
print(a[3])

a[2:]
a[1:4]

#빈 어레이 체우기
b = np.empty(3)
b
b[:] = [1,2,3]
b
b[2]

vec1 = np.array([1,2,3,4,5])
vec1
np.arange(10)
np.arange(50,100)
np.arange(0,10,0.5)

np.linspace(0,1,5)
np.linspace(0,5,5)
np.linspace(0,1,5, endpoint=False)

np.repeat(3,5)
np.repeat(vec1, 5)

np.arange(-100, 1, 1)
-np.arange(0, 100)

vec1 = np.arange(5)
np.repeat(vec1, 3)
np.tile(vec1, 3)

vec1 *3
vec1 + vec1
max(vec1)
sum(vec1)

# 35672 이하 홀수들의 합은?
sum(np.arange(1,35672,2))
#혹은
np.arange(1,35672,2).sum()
x = np.arange(1,35672,2)
sum(x)

# 2차원 배열
b = np.array([[1, 2, 3], [4, 5, 6]])
length = len(b) # 첫 번째 차원의 길이
shape = b.shape # 각 차원의 크기
size = b.size # 전체 요소의 개수
length, shape, size

a = np.array([1,2])
b = np.array([1,2,3,4])
a + b

# 벡터 생성
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
# 벡터 간 덧셈
add_result = a + b

b == 3

# 35672 보다 작은 수 중에서 7로 나눠서 나머지가 3인 숫자들의 갯수는
my = np.arange(0,35672)
result = my%7 == 3
sum(result)

