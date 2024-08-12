# 자리 바꾸기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

np.random.seed(20240729)
ord_seat = np.arange(1, 29)
#1~28 숫자 중에서 중복 없이 28개 숫자를 뽑는 방법
new_seat = np.random.choice(ord_seat, 28, replace = False)
result = pd.DataFrame({"ord_seat" : ord_seat,
                       "new_seat" : new_seat})
# result.to_csv("result.csv")
# 파이썬 기본 함수, 패키지 함수, 메서드 함수_괄호 없는거.

### 질문
# y=2x 그리기
x = np.linspace(0, 8, 2) #0과 8 사이에 2개가 되게 만들어줘.
y = 2 * x
plt.plot(x, y)
plt.show()
plt.clf()

# y= x^2 를 점 3개를 사용해서 그리기
x = np.linspace(0, 8, 3) #0과 8 사이에 2개가 되게 만들어줘.
y = x ** 2
plt.plot(x, y)
plt.show()
plt.clf()

# 조금 더 곡선처럼 보이려면?
x = np.linspace(-8, 8, 100) #0과 8 사이에 2개가 되게 만들어줘.
y = x ** 2
plt.plot(x, y)
plt.show()
plt.clf()

# 교제 57페이지 연습문제
# 2)
x = np.array([79.1, 68.8, 62.0, 74.4, 71.0, 60.6, 98.5, 86.4, 73.0, 40.8, 61.2, 68.7, 61.6, 67.7, 61.7, 66.8])
x.mean() # 포본평균균
n = len(x)
sigma = 6
# 1-알파: 0.9(신뢰수준)
alpha = 0.1
# Za/2 = Z0.05(정규분포(뮤=0, 시그마제곱 = 1)에서 상위 5%에 해당하는 x값.)
# => norm.ppf(0.95, loc = 0, scale = 1)
z_095 = norm.ppf(0.95, loc = 0, scale = 1)
z_005 = norm.ppf(0.05, loc = 0, scale = 1)

x.mean() + z_005 * 6 / np.sqrt(16)
x.mean() + z_095 * 6 / np.sqrt(16)
# 66.42~71.36

# 표본분산 설명 중
# 데이터로부터 E[X^2] 구하기기
x = norm.rvs(loc=3, scale=5, size = 10000)
np.mean(x**2)

# 표본 10만개를 추출해서 S^2를 구해보세요.
np.random.seed(20240729)
x = norm.rvs(loc = 3, scale = 5, size = 100000)
x_bar = x.mean()
sum((x - x_bar)**2) / (len(x)-1) #표본 분산 구하기.
# np.var(x) # n으로 나누기
np.var(x, ddof = 1) # n-1로 나누기 (표본 분산)

# n-1 vs n
x = norm.rvs(loc = 3, scale = 5, size = 10)
np.var(x)
np.var(x, ddof = 1)

# 파이썬 교재 8장. 122p
# 라인 그래프

economics = pd.read_csv("data/economics.csv")
economics.head()
economics.info()
sns.lineplot(data = economics, x = "date", y = "unemploy", linewidth = 0.5)
plt.show()
plt.clf()
# 날짜가 겹쳐 보임 ㅠㅠ

# 날짜 변수 부여하기.
economics["date2"] = pd.to_datetime(economics["date"])
economics.info()
economics["date2"].dt.year #어트리뷰트 함수라서 괄호 안 씀 ㅋㅋ
economics["date2"].dt.month
economics["date2"].dt.month_name() #이거는 메서드.
economics["date2"].dt.quarter

economics["quarter"] = economics["date2"].dt.quarter # 분기 추가
economics[["date2", "quarter"]]
economics["date2"].dt.day_name() #요일도 알 수 있따!
economics["date2"] + pd.DateOffset(days=3) #일 수를 더할 수 있다.
economics["date2"] + pd.DateOffset(days = 30) #30일 더해지는거.
economics["date2"] + pd.DateOffset(months = 1)
economics["year"] = economics["date2"].dt.year
economics["date2"].dt.is_leap_year.head(12) #윤년 체크

# 라인플롯롯
sns.lineplot(data = economics, x = "
year", y = "unemploy")
plt.show() # 신뢰구간 까지 나와버림림
plt.clf()

sns.lineplot(data = economics, x = "year", y = "unemploy", errorbar = None)
plt.show()
plt.clf()

sns.scatterplot(data = economics, x = "year", y = "unemploy")
plt.show()
plt.clf()

# 신뢰구간을 우리가 계산해서 해보기.
my_df = economics.groupby('year', as_index = False) \
                 .agg(
                        mon_mean = ("unemploy", "mean"),
                        mon_std = ("unemploy", "std"),
                        mon_n = ("unemploy", "count"))
my_df["left_ci"] = my_df["mon_mean"] - (1.96 * my_df["mon_std"] / np.sqrt(my_df["mon_n"]))
my_df["right_ci"] = my_df["mon_mean"] + (1.96 * my_df["mon_std"] / np.sqrt(my_df["mon_n"]))
my_df

# 우리가 계산한거로 시각화 해보기
x = my_df["year"]
y = my_df["mon_mean"]
plt.plot(x, y, color = "black")
plt.scatter(x, my_df["left_ci"], color = "blue", size = 5)
plt.scatter(x, my_df["right_ci"], color = "red", size = 5)
plt.show()
plt.clf()


### 저번에 캐글 집 가격 했던거 다시
# 저번엔 집 값을 전체 평균으로 삽입했다.
# 이번엔 집 지어진 년도끼리 평균을 내서 그거로 예상하기.

# 캐글 데이터 불러오기
house_train = pd.read_csv("data/houseprice/train.csv")
house_train.info()
house_train = house_train[["Id", "YearBuilt", "SalePrice"]] #사용할 것만 뺐다.
house_train.info()

# 연도별 평균
house_mean = house_train.groupby("YearBuilt", as_index = False) \
                        .agg(mean_year = ("SalePrice", "mean"))
house_mean

# 테스트 데이터 불러오기
house_test = pd.read_csv("data/houseprice/test.csv")
house_test = house_test[["Id", "YearBuilt"]]

# 연도별 평균을 테스트 데이터에 컬럼 추가하기
house_test = pd.merge(house_test, house_mean, how = "left", on = "YearBuilt")
house_test = house_test.rename(columns = {"mean_year" : "SalePrice"})
house_test

# 입력 안 된거 확인.
sum(house_test["SalePrice"].isna())
house_test.loc[house_test["SalePrice"].isna()]

# 전체 평균
price_mean = house_train["SalePrice"].mean()

# 비어 있는 곳 체우기기
house_test["SalePrice"] = house_test["SalePrice"].fillna(price_mean) # 파이썬 교제 184p

# 다시 확인하기
sum(house_test["SalePrice"].isna())

# 제출할 데이터 불러오기
sub_df = pd.read_csv("data/houseprice/sample_submission.csv")
sub_df

# 값 바꿔치기
sub_df["SalePrice"] = house_test["SalePrice"]

# 내보내기
# sub_df.to_csv("data/houseprice/sample_submission_240729.csv", index = False)

####### 각자 변수 3개 이하 선택.######
# 나는 지어진 연도
# GarageCars : 차량 수용 인원에 따른 차고 크기

# 캐글 데이터 불러오기
house_train = pd.read_csv("data/houseprice/train.csv")
house_train.info()
house_train = house_train[["Id", "YearBuilt", "GarageCars", "SalePrice"]]
house_train.info()

# 연도별, 차고별 평균
house_mean = house_train.groupby(["YearBuilt", "GarageCars"], as_index = False) \
                        .agg(mean_year = ("SalePrice", "mean"))
house_mean

# 테스트 데이터 불러오기
house_test = pd.read_csv("data/houseprice/test.csv")
house_test = house_test[["Id", "YearBuilt", "GarageCars"]]
house_test

# 연도별 평균을 테스트 데이터에 컬럼 추가하기
house_test = pd.merge(house_test, house_mean, how = "left", on = ["YearBuilt", "GarageCars"])
house_test = house_test.rename(columns = {"mean_year" : "SalePrice"})

# 입력 안 된거 확인.
sum(house_test["SalePrice"].isna())
house_test.loc[house_test["SalePrice"].isna()]

# 전체 평균
price_mean = house_train["SalePrice"].mean()

# 비어 있는 곳 체우기기
house_test["SalePrice"] = house_test["SalePrice"].fillna(price_mean) # 파이썬 교제 184p

# 다시 확인하기
sum(house_test["SalePrice"].isna())

# 제출할 데이터 불러오기
sub_df = pd.read_csv("data/houseprice/sample_submission.csv")
sub_df

# 값 바꿔치기
sub_df["SalePrice"] = house_test["SalePrice"]

# 내보내기
# sub_df.to_csv("data/houseprice/sample_submission_240729-2.csv", index = False)
