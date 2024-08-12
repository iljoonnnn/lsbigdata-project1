#데이터타입
x = 15.55
print(x, "는 ", type(x), "형식입니다.", sep=' ')

# 문자형 데이터 예제
a = "Hello, world!"
b = 'python programming'

print(a, type(a))
print(b, type(b))

# 여러 줄 문자열
ml_str = """This is
a multi-line
string"""
print(a, type(a))
print(b, type(b))
print(ml_str, type(ml_str))

# 문자열 결합
greeting = "안녕" + " " + "파이썬!"
print("결합 된 문자열:", greeting)
# 문자열 반복
laugh = "하" * 3
print("반복 문자열:", laugh)

###리스트 구조

#리스트
#리스트는 순서가 있다. 여러가지 타입을 리스트안에 넣을 수 있다.
fruit = ["apple", "banana", "chery"]
type(fruit)

numbers = [1,2,3,4,5]
type(numbers)

mixed_list = [1, "Hello", [1, 2, 3]]
type(mixed_list)

#튜플
#리스트랑 비슷하다. 생성후에는 수정할 수 없다. 대신 실행이 빠르다!!
a = (10, 20, 30, 40, 50) # a = 10, 20, 30 과 동일
b_int = (42)
type(b_int)
b_tp = (42,)
type(b_tp)
print("좌표:", a)
print("단원소 튜플:", b_tp)
b_int=10
b_int
b_tp=10
b_tp

a[0] #[]어떤 객체 앞에 대괄호를 붙이면 그 요소에 접근할 수 있다.
a[1]
a[2]

a_list = [10,20,30]
a_list[1]
a_list[1] = 25
a_list

a_tp = (10,20,30)
a_tp[1]
a_tp[1] = 25
a_tp

print("마지막 두개 좌표:", a[2:])

a[3:] # 해당 인덱스 이상
a[:3] # 해당 인덱스 미만
a[2:4] # 해당 인덱스 이상 & 미만

a_ls=[10,20,30,40,50]
a_ls[1:4]

#사용자 정의함수수
def min_max(numbers):
  return min(numbers), max(numbers)

a=[1,2,3,4,5]
min_max(a)
result = min_max(a)

#딕셔너리
iljoon = {
  "name": "iljoon",
  "age": (26,),
  "livein": ["서울","건대"]
}
print(iljoon)

iljoon_livein = iljoon.get('livein')
iljoon_livein[0]
#일단 실행을 한다. 이후 나오는 결과 값을 본뒤 타입을 파악. 이후 그것만 사용할 방법을 생각해본다.

###집합합
fruits = {'apple', 'banana', 'cherry', 'apple'}
type(fruits)
print("Fruits set:", fruits) # 중복 'apple'은 제거됨

empty_set = set()
print("Empty set:", empty_set)

empty_set.add("apple")
empty_set.add("banana")
empty_set
empty_set.remove("banana")
empty_set.discard("banana")

# 집합 간 연산
other_fruits = {'berry', 'cherry'}
union_fruits = fruits.union(other_fruits)
intersection_fruits = fruits.intersection(other_fruits)
print("Union of fruits:", union_fruits)
print("Intersection of fruits:", intersection_fruits)

p = True
q = False
print(p, type(p))
print(q, type(q))
print(p + p)

age = 10
is_active = True
is_greater = age > 5 # True 반환
is_equal = (age == 5) # False 반환
print("Is active:", is_active)
print("Is 10 greater than 5?:", is_greater)
print("Is 10 equal to 5?:", is_equal)

#조건문
a=3
if (a == 2):
  print("a는 2와 같습니다.")
else:
  print("a는 2와 같지 않습니다.")

#문자열로 바꾸기기
num = 123
str_num = str(num)
print("문자열:", str_num, type(str_num))

# 문자열형을 숫자형(실수)으로 변환
num_again = float(str_num)
print("숫자형:", num_again, type(num_again))

# 리스트와 튜플 변환
lst = [1, 2, 3]
print("리스트:", lst)
tup = tuple(lst)
print("튜플:", tup)

set_example = {'a', 'b', 'c'}
# 딕셔너리로 변환 시, 일반적으로 집합 요소를 키 또는 값으로 사용
dict_from_set = {key: True for key in set_example}
print("Dictionary from set:", dict_from_set)

bool(0)
bool(7)

int(True)
int(False)

str(True)
str(False)

bool("True")
bool("False")

##############################
import seaborn as sns
import matplotlib.pyplot as plt

var = ['a', 'a', 'b', 'c']
var

sns.countplot(x=var)
plt.show()
plt.clf()
df = sns.load_dataset("titanic")
sns.countplot(data=df, x="sex")
plt.show()
plt.clf()

sns.countplot(data=df, x="class", hue = "alive")
plt.clf()
sns.countplot(data=df, y="class", hue = "alive")
?sns.countplot
plt.clf()
sns.countplot(data=df, x="sex", hue="sex")
plt.show()

sns.countplot(data=df,
              y="class",
              hue="alive",
              orient="v")
plt.show()

#!pip install scikit-learn
import sklearn.metrics

sklearn.metrics.accuracy_score()
from sklearn import metrics
metrix.accuracy_score()

import sklearn.metrics as met
