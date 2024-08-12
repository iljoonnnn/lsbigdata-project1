import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from scipy.stats import t

# 데이터 불러오기
train = pd.read_csv("data/houseprice/train.csv")
test = pd.read_csv("data/houseprice/test.csv")
sample_submission = pd.read_csv("data/houseprice/sample_submission.csv")

x = train[["GrLivArea", "GarageArea"]]
y = np.array(train["SalePrice"])

model = LinearRegression()

model.fit(x, y)

model.coef_         # 기울기 a
model.intercept_    # 절편 b
slope = model.coef_
intercept = model.intercept_

y_pred = model.predict(x)

test_x = test[["GrLivArea", "GarageArea"]]
y_pred = model.predict(test_x) #에러 뜸. 결측치 있어서...

# 결측치 확인.
test_x["GrLivArea"].isna().sum()
test_x["GarageArea"].isna().sum() #결측치 있음.
test_x = test_x.fillna(test["GarageArea"].mean())

y_pred = model.predict(test_x)

sample_submission["SalePrice"] = y_pred #셈플에 가격 넣기
# ================ # 여기까지가 지난주에 한거 =======================


########## 이번엔 변수 3개를 해보자. 아니. 모든 숫자 변수를 써보자. ###########
train = pd.read_csv("data/houseprice/train.csv")
test = pd.read_csv("data/houseprice/test.csv")
sample_submission = pd.read_csv("data/houseprice/sample_submission.csv")

# 숫자형 변수만 선택하기
train_x = train.select_dtypes(include = [int, float])
train_x = train_x.iloc[:, 1:-1] # id와 saleprice 컬럼 빼고 모두 선택
train_y = np.array(train["SalePrice"])
train_x.isna().sum() # 결측치가 있음. LotFrontage, MasVnrArea, GarageYrBlt

# 결측치 제거
train_x = train_x.dropna(subset = ["LotFrontage", "MasVnrArea", "GarageYrBlt"])
train_x.isna().sum() # 결측치가 없어짐!

model = LinearRegression()

model.fit(train_x, train_y) # x와 y 길이가 달라서 에러 뜸. y에서도 같은 행 제거하자.

### 변수 다시 깔끔하게 선택.
train_x = train.select_dtypes(include = [int, float])
train_x = train_x.iloc[:, 1:]
train_x = train_x.dropna(subset = ["LotFrontage", "MasVnrArea", "GarageYrBlt"])
train_y = train_x[["SalePrice"]]
train_x = train_x.iloc[:, :-1]

model = LinearRegression()

model.fit(train_x, train_y)

model.coef_
model.intercept_
slope = model.coef_
intercept = model.intercept_

train_y_pred = model.predict(train_x)

# test 데이터를 불러와!
test_x = test.select_dtypes(include = [int, float])
test_x = test_x.iloc[:, 1:]

# test_y_pred = model.predict(test_x) # 에러 뜸. 여기도 결측치가 있구나?!

# test_x의 결측치를 평균으로 바꾸기.
test_x.isna().sum()
test_x = test_x.fillna(test[["LotFrontage", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "GarageYrBlt", "GarageCars", "GarageArea"]].mean())

test_y_pred = model.predict(test_x)

sample_submission["SalePrice"] = test_y_pred #셈플에 가격 넣기
# sample_submission.to_csv("sample_submission_240805.csv", index = False)
# 0.68018 나옴 ㅠㅠ

################ 이번엔 train 모델 만들 때 평균으로 대체해서 사용해보자.
train = pd.read_csv("data/houseprice/train.csv")
test = pd.read_csv("data/houseprice/test.csv")
sample_submission = pd.read_csv("data/houseprice/sample_submission.csv")

train_x = train.select_dtypes(include = [int, float])
train_x = train_x.iloc[:, 1:-1]
train_x.isna().sum()
train_x = train_x.fillna(train[["LotFrontage", "MasVnrArea", "GarageYrBlt"]].mean())
train_x.isna().sum()
train_y = train["SalePrice"]

model = LinearRegression()

model.fit(train_x, train_y)

model.coef_
model.intercept_

train_y_pred = model.predict(train_x)

test_x = test.select_dtypes(include = [int, float])
test_x = test_x.iloc[:, 1:]
test_x.isna().sum()
test_x = test_x.fillna(test[["LotFrontage", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "GarageYrBlt", "GarageCars", "GarageArea"]].mean())
test_x.isna().sum()

test_y_pred = model.predict(test_x)

sample_submission["SalePrice"] = test_y_pred #셈플에 가격 넣기
#sample_submission.to_csv("sample_submission_240805-1.csv", index = False)
# 0.22086 나옴.

#############################################################################
#############################################################################
# 이번엔 train 데이터를 평균치, 최빈값으로 대체 + 딕셔너리 사용 + 평균 채워넣는거 쉽게.효은님 아이디어
train = pd.read_csv("data/houseprice/train.csv")
test = pd.read_csv("data/houseprice/test.csv")
sample_submission = pd.read_csv("data/houseprice/sample_submission.csv")

train_x = train.select_dtypes(include = [int, float])
train_x = train_x.iloc[:, 1:-1]
train_x.isna().sum()
train_x_fill_values = {
    "LotFrontage" : train_x["LotFrontage"].mean(),
    "MasVnrArea" : train_x["MasVnrArea"].mean(),
    "GarageYrBlt" : train_x["GarageYrBlt"].mean()
}
train_x = train_x.fillna(value = train_x_fill_values) # 딕셔너리 방법
train_x.isna().sum()

train_y = train["SalePrice"]

model = LinearRegression()

model.fit(train_x, train_y)

train_y_pred = model.predict(train_x)


test_x = test.select_dtypes(include = [int, float])
test_x = test_x.iloc[:, 1:]
test_x.isna().sum()
test_x = test_x.fillna(test_x.mean()) # 쉽게 평균으로 바꿀 수 있음.
test_x.isna().sum()

test_y_pred = model.predict(test_x)

sample_submission["SalePrice"] = test_y_pred #셈플에 가격 넣기
# sample_submission.to_csv("data/houseprice/sample_submission_240805-4.csv", index = False)
# 0.22086 나옴.


#############################################################################
#############################################################################

# 직선 그려보기
# y = 2x + 3
from scipy.stats import norm

x = np.linspace(0, 100, 400) # 모직선
y = 2 * x + 3
plt.plot(x, y, color = "black")
plt.show()
plt.clf()


np.random.seed(20240805)

obs_x = np.random.choice(100,20) # 집 넓이라고 가정.
epsilon_i = norm.rvs(loc=0, scale=20, size = 200) # 점을 만들 때 퍼짐 정도 정해주기.
obs_y = 2 * obs_x + 3 + epsilon_i

# 그래프 그리기
plt.plot(x, y, label = 'y = 2x + 3', color = "black")
plt.scatter(obs_x, obs_y, color = "blue", s=3)
plt.show()
#plt.clf()

model = LinearRegression()
obs_x=obs_x.reshape(-1,1)
model.fit(obs_x, obs_y)

coef = model.coef_[0]           # 기울기 a
intercept = model.intercept_    # 절편 b

my_x = np.arange(0, 100, 5)
my_y = coef * my_x + intercept

plt.plot(my_x, my_y, color = "red")
plt.show()
plt.clf()

#!pip install statsmodels
import statsmodels.api as sm # 싸이파이와 다른 패키지. 선형회귀모델 가능함

obs_x = sm.add_constant(obs_x)
model = sm.OLS(obs_y, obs_x).fit()
print(model.summary())

#########################
##########################
#########################

# 귀무가설 대립가설

8.79 / np.sqrt(20)

# 모 표준편차가 8.79
# 모평균이 10
# 셈플 20개 뽑아서 18이 나옴
# 이 때 모평균이 10이라는 말이 맞냐?

# 18이상이 나올 확률
1 - norm.cdf(18, loc=10, scale = 1.96)
# 여기서 구한 0.0001이 p-value, 유의확률 이라고 한다.

### 교제 57페이지 자동차 문제
myarray = np.array([15.078, 15.752, 15.549, 15.56, 16.098, 13.277, 15.462, 16.116, 15.214, 16.93, 14.118, 14.927, 15.382, 16.709, 16.804])

# 2. 검정을 위한 가설을 명확하게 서술하시오.
'''
h0(귀무가설): 슬통 자동차는 에너지 소비 효율이 16.0 이상이다.
hA(대립가설): 슬통 자동차는 에너지 소비 효율이 16.0 미만이다.
'''

# 3. 검정통계량 계산하시오.
# t = {x_bar - mu0} / {S / root(n)}
x_bar = myarray.mean() # 표본 평균
mu_zero = 16.0 # 귀무가설에서 주장하는 평균 값
my_s = np.std(myarray, ddof=1) # n-1로 나눠진 분산의 제곱근 s
my_n = len(myarray) # n

t_value = (x_bar - mu_zero) / (my_s / np.sqrt(my_n))
t_value # -1.85

# 4. p‑value을 구하세요. (유의확률)
p_value = t.cdf(x = t_value, df = my_n-1)
p_value #0.042
# 이 말은 0.042 확률로 귀무가설을 기각하고 대립가설을 채택하는 것이다? 맞나요?

# 6. 현대자동차의 신형 모델의 평균 복합 에너지 소비효율에 대하여 95% 신뢰구간을 구해보세요.
x_bar + my_s / np.sqrt(my_n) # 15.784
x_bar - my_s / np.sqrt(my_n) # 15.278
# 15.278 ~ 15.784


