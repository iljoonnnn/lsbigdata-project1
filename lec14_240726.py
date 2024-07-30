import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

#X ~ U(2,6)
# 시작점, 끝점이 모수.
from scipy.stats import uniform
uniform.rvs(loc = 2, scale = 4, size=1)

# X ~균일분포(a,b)
# 파이썬 함수에서는 loc은 시작점. scale은 길이
# loc: a    scale = b-a

x = np.linspace(0, 8, 100)
y = uniform.pdf(x, loc = 2, scale = 4)
plt.scatter(x, y)
plt.show()
plt.clf()
plt.plot(x, y)
plt.show()
plt.clf()

# p(x<3.25) =?
uniform.cdf(3.25, loc = 2, scale = 4)

# p(5<x<8.39) =?
uniform.cdf(8.39, loc = 2, scale = 4) - uniform.cdf(5, loc = 2, scale = 4)

# 상위 7% 값은?
uniform.ppf(0.93, loc = 2, scale = 4)

# 표본 20개를 뽑아서 표본 평균도 계산해보세요!
uniform.rvs(loc = 2, scale =4, size = 20).mean()
# 우리들의 평균값이 다르다. 모두 같은 값을 뽑게 하려면 random_state 옵션
uniform.rvs(loc = 2, scale = 4, random_state=42).mean()
# 나중에 다같이 넘파이 시드 설정으로 뽑아보자!
np.random.seed(42)
uniform.rvs(loc = 2, scale =4, size = 20).mean()

# 방금 검정색 블럭 20개 뽑아서 파란색 블럭 하나를 뽑았다.
x = uniform.rvs(loc = 2, scale = 4, size = 20* 1000, random_state=42) # 검정 블록 2만개 뽑기.
x.shape
x = x.reshape(-1, 20) # 검정 블록 1000 * 20개로 나눔.
x.shape
blue_x = x.mean(axis = 1) # 평균 구하여 파란색 블럭으로 변경경
blue_x.shape
sns.histplot(blue_x, stat = "density")
plt.show()
# 여기에 정규분포 선을 그리고 싶다!
min(blue_x)
max(blue_x)
mean_blue_x = np.mean(blue_x)
sigma_blue_x = np.sqrt(sum((blue_x - mean_blue_x)**2) / len(blue_x))
xx = np.linspace(3,5,100)
yy = norm.pdf(xx, loc = mean_blue_x, scale = sigma_blue_x)
plt.plot(xx, yy)
plt.show() #잘 안되군... # 잘 됐따! ㅋㅋㅋㅋ
plt.clf()

# X bar ~ N(mu, sigma^2/n)
# X bar ~ N(4, 1.33333/20)
# 분산 구하는 명령어
uniform.var(loc =2, scale = 4)
uniform.expect(loc = 2, scale = 4) #이거는 기대값 구하는 명령어
xmin = min(blue_x)
xmax = max(blue_x)
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc=4, scale = np.sqrt(1.33333/20)) #높이 알기
plt.plot(x_values, pdf_values, color = "red", linewidth = 2)
plt.show()

## 신뢰구간
# 중간이 95%인 경계 구하기
a = norm.ppf(0.025, loc = 4, scale = np.sqrt(1.33333/20))
b = norm.ppf(0.975, loc = 4, scale = np.sqrt(1.33333/20))

norm.ppf(0.025, loc = 0, scale = 1)
norm.ppf(0.975, loc = 0, scale = 1)
# 신뢰구간 95%이면 평균 쁠마 1.96 시그마

# 중간이 99%인 경계 구하기
c = norm.ppf(0.005, loc = 4, scale = np.sqrt(1.33333/20))
d = norm.ppf(0.995, loc = 4, scale = np.sqrt(1.33333/20))

norm.ppf(0.005, loc = 0, scale = 1)
norm.ppf(0.995, loc = 0, scale = 1)
# 신뢰구간 99%이면 평균 쁠마 2.57 시그마

plt.axvline(a)
plt.axvline(b)
plt.axvline(c)
plt.axvline(d)
plt.show()
plt.clf()
==========================================================
### 정규분포 선 그리기
# X bar ~ N(mu, sigma^2/n)
# X bar ~ N(4, 1.33333/20)
x_values = np.linspace(3, 5, 100)
pdf_values = norm.pdf(x_values, loc=4, scale = np.sqrt(1.33333/20)) #높이 알기
plt.plot(x_values, pdf_values, color = "red", linewidth = 2)

# 표본평균(파란벽돌) 점 찍기
blue_x = uniform.rvs(loc = 2, scale = 4, size = 20).mean()
plt.scatter(blue_x, 0.002, color = "blue", zorder=10, s=10)

# 표본평균을 통하여 유추하는 모평균. 의 범위를 구하기
a = blue_x + 0.665
b = blue_x - 0.665
plt.axvline(x=a, color="blue", linestyle = "--", linewidth = 1)
plt.axvline(x=b, color="blue", linestyle = "--", linewidth = 1)

# 기대값 표현
plt.axvline(x= 4, color="green", linestyle = '-')

plt.show()
plt.clf()
==========================================================

# 평균이 0이고 표준편차가 1인 정규분포 = 표준정규분포
norm.ppf(0.025, loc = 0, scale = 1)
norm.ppf(0.975, loc = 0, scale = 1) #1.96
# 그러면 95%의 신뢰구간은 평균 + 1.96 표준편차
#                              - 1.96 표준편차









