from scipy.stats import bernoulli
from scipy.stats import binom
from scipy.stats import norm

# 확률질량함수(pmf)
# 확률 변수가 갖는 값에 해당하는 확률을 저장하고 있는 함수
# bernoulli.pmf(k, p)
# P(X = 1)
bernoulli.pmf(1, 0.3)
# P(X = 0)
bernoulli.pmf(0, 0.3)



from scipy.stats import binom
# 이항분포 X ~ P(X = k | n, p)
# n: 베르누이 확률 변수 더한 갯수
# p: 1이 나올 확률
# binom.pmf(k, n, p)
binom.pmf(0, n=2, p=0.3) #확률 알려줌.
binom.pmf(1, n=2, p=0.3)
binom.pmf(2, n=2, p=0.3)
# 0/1이 나올 수 있고 확률이 각각 0.7/0.3인 확률변수를 두 개 더해서 만든 확률 변수

# X ~ B(n, p)
# 베르누이 함수가 30개이면??
import numpy as np
probability = binom.pmf(np.arange(31), n=30, p=0.3)
# 위에 명령어 해석. 1이 나올 확률이 0.3인 베르누이 함수를 30개 합친거에서 0이 나올 확률, 1이 나올 확률... 다 뽑는거.
probability[0]
probability[30] #확률이 0.3이어서 다름 ㅠㅠ



===============================================
#팩토리얼 어쩌구 계산
math.factorial(54) / math.factorial(26) / math.factorial(28)


n = 54
r = 26
n_pac_array = np.arange(1, n+1)
r_pac_array = np.arange(1, r+1)
n_minus_r_pac_array = np.arange(1, (n-r+1))
# 원소끼리 모두 곱하는 명령을 몰라서 이 방법 취소 ㅠㅠ
===============================================

import math
# math.comb(a, b)
# a개 중에 b개를 선택할 경우의 수 나누기 모든 경우의 수.
math.comb(2,0) * 0.3**0 * (1-0.3)**2
math.comb(2,1) * 0.3**1 * (1-0.3)**1
math.comb(2,2) * 0.3**2 * (1-0.3)**0
# p가 0.3이라서 위같이 된거임.

#위의 식을 더 쉽게 해주는거.
# probability mass function (확률 질량 함수수)
binom.pmf(0, 2, 0.3)
binom.pmf(1, 2, 0.3)
binom.pmf(2, 2, 0.3)

## 연습.
# X ~B(n=10, p=0.36)
# P(X = 4) =?
#    X가 베르누이 함수 10개를 합치고 확률 기준점(p)는 0.36이다.
#    이 때 4가 나올 확률은?
binom.pmf(4, 10, 0.36)

## 연습2
# 같은 확률변수이다.
# P(X = 4<=4) =?
np.arange(5)
binom.pmf(np.arange(5), 10, 0.36).sum()

#아니면? 아래에서 배울 cdf 사용해보자.
binom.cdf(4, 10, 0.36)

## 연습 3
# p(2<x<=8) 확률은?
np.arange(3,9)
binom.pmf(np.arange(3,9), 10, 0.36).sum()

## 연습 4
# X ~B(30, 0.2)
# 확률변수 x가 4보다 작고 25보다 크거나 같을 확률
np.arange(4)
a = binom.pmf(np.arange(4), 30, 0.2).sum()

np.arange(25,31)
b = binom.pmf(np.arange(25,31), 30, 0.2).sum()
a+b

# 다른 방법
np.arange(4,25)
1 - (binom.pmf(np.arange(4,25), 30, 0.2).sum())

# rvs 함수 (random variates sample)
# 표본 추출 함수
# X1 ~ Bernulli(p=0.3)
bernoulli.rvs(p=0.3)
# X2 ~ Bernulli(p=0.3)
bernoulli.rvs(p=0.3)
bernoulli.rvs(p=0.3) + bernoulli.rvs(p=0.3)
# X ~ B(n=2, p=0.3)
binom.rvs(n = 2, p = 0.3)

# X ~ B(30, 0.26)
# 표본 30개를 뽑아보세요!
binom.rvs(n = 30, p = 0.26, size = 30)
# 위 함수의 기댓값은?
aa = np.arange(31)
bb = binom.pmf(np.arange(31), n = 30, p = 0.26)
sum(aa * bb)
# 이거는 혼자 코드 짜보자. 성공!

# 위 함수 시각화
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
x = np.arange(31) #나올 수 있는 수
prob_x = binom.pmf(np.arange(31), 30, 0.26)# 각각의 확률
my_df = pd.DataFrame({'x' : x, 'probability' : prob_x})
sns.barplot(data = my_df, x = 'x', y = 'probability')
plt.show()
plt.clf()

#선생님 방식
x = np.arange(31)
prob_x = binom.pmf(x, 30, 0.26)
df = pd.DataFrame({"x" : x, "prob" : prob_x})
df
sns.barplot(data = df, x = 'x', y = 'prob')
plt.show()
plt.clf()
# p가 0.26인 베르누이 분포 30개를 합한 것의 확률 질량 함수를 나타낸 것이다.

# cdf: cumulative dist. finction
# (누적 확률분포 함수)
# F(X=x) = P(X <= x)
binom.cdf(4, n=30, p=0.26)

# 연습
# p(4<x<=18)
# p(x<=18) - p(x<=4)
binom.cdf(18, n=30, p=0.26) - binom.cdf(4, n=30, p=0.26)

# p(13< x < 20)
binom.cdf(19, n=30, p=0.26) - binom.cdf(13, n=30, p=0.26)

# 다시 그래프
x = np.arange(31)
prob_x = binom.pmf(x, 30, 0.26)
df = pd.DataFrame({"x" : x, "prob" : prob_x})
df
sns.barplot(data = df, x = 'x', y = 'prob')
# Add a point at (2,0)
x_1 = binom.rvs(n = 30, p = 0.26, size = 1) #임의의 점점
x_1
plt.scatter(x_1, 0.002, color = "red", zorder = 10, s=5) #임의로 점 찍기
plt.show()
plt.clf()

# 그래프 변형
x = np.arange(31)
prob_x = binom.pmf(x, 30, 0.26)
df = pd.DataFrame({"x" : x, "prob" : prob_x})
df
sns.barplot(prob_x)
# Add a point at (2,0)
x_1 = binom.rvs(n = 30, p = 0.26, size = 10) #임의의 점점
x_1
plt.scatter(x_1, np.repeat(0.002, 10), color = "red", zorder = 10, s=5) #임의로 점 찍기
plt.axvline(7.8, color = "green", linestyle='--', linewidth=2)
plt.axvline(x_1.mean(), color = "red", linestyle='-', linewidth=2)
plt.show()
plt.clf()

# ppf
binom.ppf(0.5, n=30, p=0.26)
binom.cdf(8, n=30, p=0.26)
binom.cdf(7, n=30, p=0.26)

# 연습 문제 p(x<?) = 0.7
binom.ppf(0.7, n=30, p=0.26)
binom.cdf(8, n=30, p=0.26) #0.7확률이면 8까지는 포함된다는 이야기네.
binom.cdf(9, n=30, p=0.26)


1/np.sqrt(2* math.pi)
from scipy.stats import norm
norm.pdf(0, loc=0, scale = 1)

# 뮤=3, 시그마=4, x=5일 때 몇?
norm.pdf(5, 3, 4) #그냥 콤마로 되네?

# 우리는 pmf 함수를 통해서 어떤 값을 넣었을 때 확률을 줬다.
# 이걸 이용해서 그림을 그렸었다.

x = np.linspace(-3, 3, 5) #-3에서 3 사이의 숫자 5개를를 알려줘
y = norm.pdf(x, 0, 1) # x가 위에 어레이, 평균이 0, 분산이 1일 때 확률을 알려줘.

plt.scatter(x, y, color = "red")
plt.show()
plt.clf()

# 더 촘촘하게게
x = np.linspace(-3, 3, 100)
y = norm.pdf(x, loc = 0, scale = 1) # x가 위에 어레이, 평균이 0, 분산이 1일 때 확률을 알려줘.
plt.scatter(x, y, color = "red", s=1)
plt.show()
plt.clf()

# 그냥 플랏으로
x = np.linspace(-5, 5, 100) #범위가 늘어나면서 좁혀진것처럼 보인다.
y = norm.pdf(x, loc = 0, scale = 1)
plt.plot(x, y)
plt.show()
plt.clf()

## mu (loc) 가 변한다는건 뭐가 변한다는걸까?
x = np.linspace(-5, 5, 100)
y = norm.pdf(x, loc = 3, scale = 1) #mu를 3으로로
plt.plot(x, y)
plt.show() #중심이 변한다!
plt.clf()

## sigma (scale): 분포의 퍼짐 결정하는 모수 #시그마는 표준편차이다.
# 표준편차 제곱은 분산.
# 모수란? 분포의 특징을 결정하는 수
x = np.linspace(-5, 5, 100)
y = norm.pdf(x, loc = 0, scale = 1)
y2 = norm.pdf(x, loc = 0, scale = 2)
y3 = norm.pdf(x, loc = 0, scale = 0.5)
plt.plot(x, y, color = "black")
plt.plot(x, y2, color = "red")
plt.plot(x, y3, color = "blue")
plt.show()
plt.clf()
# 

norm.cdf(0, loc = 0, scale = 1) #왼쪽 끝에서 0까지의 넓이(확률)은?
norm.cdf(100, loc = 0, scale = 1) #왼쪽 끝에서 100까지의 넓이(확률)은?
#binom.cdf()랑 다른거임. 이거는 막대그래프에서 넓이(확률) 계산

# p(-2 < x < 0.54) = ?
norm.cdf(0.54, loc = 0, scale = 1) - norm.cdf(-2, loc = 0, scale = 1)

# p(x<1 or x>3) = ?
norm.cdf(1, loc = 0, scale = 1) + (1 - norm.cdf(3, loc = 0, scale = 1))

# X ~ N(3, 5^2)     _X가 평균이 3이고 분산이 5^2인 정규분포를 따를때. 라는 뜻.
# 일때는? *참고로 괄호 안에 오른쪽에는 분산이 들어가는게 국룰.
# 정보: 정규 분포의 영어 이름: Nomal distribution
norm.cdf(5, 3, 5) - norm.cdf(3, 3, 5)

# 위에 확률 변수에서 표본 1000개 뽑아보자!
myarray = norm.rvs(loc = 3, scale = 5, size = 1000)
sum((myarray > 3) & (myarray < 5))/1000

# 평균: 0, 표준편차: 1
# 표본 1000개 뽑아서 0보다 작은 비율 확인.
norm.cdf(0, 0, 1)
myarray = norm.rvs(0, 1, 1000) #뮤는 0, 표준편차는 1인 정규분포에서 표본 1000개 뽑기.
np.mean(myarray<0)

# 그래프 그리기
x = norm.rvs(loc=3, scale = 2, size = 1000)
x
sns.histplot(x, stat="density") #stat="density" 옵션을 쓰면 y축을 카운트에서 퍼센트로 바꿔준다.
#여기에 pdf 그리기
xmin = x.min()
xmax = x.max()
x_values = np.linspace(xmin, xmax, 100) #값 두개 넣은거 사이에 100개의 숫자를 만든다.
pdf_values = norm.pdf(x_values, loc=3, scale = 2) #각 값이 나올 수 있는 확률을 구한다. 그것은 나중에 높이가 된다.
plt.plot(x_values, pdf_values, color="red", linewidth=2)
plt.show()
plt.clf()


# 숙제 Qmd
# 1. 정규분포 pdf 값을 계산하는 자신만의
# 파이썬 함수를 정의하고, 
# 정규분포 mu = 3, sigma = 2 의 pdf를 그릴 것.
# scipy.stats 함수 사용 안하고 하기.
import numpy as np
import math
import matplotlib.pyplot as plt
def nomal_distribution_function(x, mu, sigma):
    result=(np.exp(1)**((((x-mu)/sigma)**2)/(-2)))/(sigma * np.sqrt(2*(math.pi)))
    return result
nomal_distribution_function(0, 3, 2)

nomal_distribution_function(0, 3, 2)
x = np.linspace(16, -10, 100)
y = nomal_distribution_function(x, 3, 2)
plt.scatter(x, y)
plt.show()
plt.clf()
# 2. 파이썬 scipy 패키지 사용해서 다음과 같은
# 확률을 구하시오.
# X ~ N(2, 3^2)
# 1) P(X < 3)
# 2) P(2 < X < 5)
# 3) P(X < 3 or X > 7)
from scipy.stats import bernoulli
from scipy.stats import binom
from scipy.stats import norm
#1
norm.cdf(3, 2, 3)
#2
norm.cdf(5, 2, 3) - norm.cdf(2, 2, 3)
#3
norm.cdf(3, 2, 3) + (1 - norm.cdf(7, 2, 3))

# 3. LS 빅데이터 스쿨 학생들의 중간고사 점수는
# 평균이 30이고, 분산이 4인 정규분포를 따른다.
# 상위 5%에 해당하는 학생의 점수는?
norm.ppf(0.95, 30, 2)
