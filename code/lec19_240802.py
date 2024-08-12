import numpy as np
from scipy.optimize import minimize
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt


# 어제 회귀선 만드는데 쓰이는 최소값 찾는 거 알아보기

# y = x^2 + 3 함수로 만들어보기
def my_f(x):
    y = x**2 +3
    return y

my_f(0)
my_f(1)
my_f(-1)

# 이 때 최소값을 찾아주는 함수: minimize
# 최소값 구하기

def my_f(x):
    y = x**2 +3
    return y

# 초기 추정값
initial_guess = [0] # 알고자 하는 함수에 두개를 입력해야 되면 [0,0] 이런식으로 적어야됨.

# 최소값 찾기
result = minimize(my_f, initial_guess) #내가 알고자 하는 함수. 괄호 빼야됨.

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)
# 미분을 이용해서 기울기가 0인 부분을 찾는 것임.

## 연습 2
def my_f2 (x):
    z = x[0]**2 + x[1]**2 + 3
    return z
# 변수가 2개일 때는 리스트를 함수에 넣는 방식으로 해야됨.

initial_guess = [0, 0]

result = minimize(my_f2, initial_guess)

print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x[0])
print("최소값을 갖는 y 값:", result.x[1])

### 연습 3  f(x,y,z) = (x-1)^2 + (y-2)^2 + (z-4)^2 +7 최솟값 구하기
def my_f3(mylist):
    return (mylist[0]-1)**2 + (mylist[1]-2)**2 + (mylist[2]-4)**2 +7

initial_guess = [0, 0, 0]
result = minimize(my_f3, initial_guess)

print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

########## 위에 한걸 서브미션에 넣고 캐글에 제출해보기 #########
train = pd.read_csv("data/houseprice/train.csv")
x = np.array(train["BedroomAbvGr"]).reshape(-1, 1)
y = np.array(train["SalePrice"])

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_         # 기울기 a
model.intercept_    # 절편 b
slope = model.coef_[0] # 어레이로 나와서 원소 뺴야됨
intercept = model.intercept_

# 식 만들기!!
predict_y = slope * x + intercept # 회귀선 구한거로 값 구하는 식임.
predict_y = model.predict(x) # 이게 사실 더 편함

# 이제 x를 테스트 데이터에서 넣으면 된다.
test = pd.read_csv("data/houseprice/test.csv")
x = np.array(test["BedroomAbvGr"]).reshape(-1, 1)
predict_y = model.predict(x) # 예측 y값이라는 뜻의 변수명

# 구한 값을 서브미션에 넣는다.
sample_submission = pd.read_csv("data/houseprice/sample_submission.csv")
sample_submission["SalePrice"] = predict_y
#sample_submission.to_csv("sample_submission_240802-3.csv", index = False)


==========================================================================
########## 이번엔 내가 변수 선택 #########
train = pd.read_csv("data/houseprice/train.csv")
# 변수 선택하기
sns.lineplot(data = train, x = train["OverallQual"], y = train["SalePrice"])
plt.show()
plt.clf()
# 우상향 하구만!!

### 회귀모델로 분석하기.

x = np.array(train["OverallQual"]).reshape(-1, 1) # 마감 퀄리티로 해보자!
y = np.array(train["SalePrice"])

model = LinearRegression()

model.fit(x, y)

model.coef_         # 기울기 a
model.intercept_    # 절편 b
slope = model.coef_[0] # 어레이로 나와서 원소 뺴야됨
intercept = model.intercept_

y_pred = model.predict(x)

# 시각화로 중간 점검!
plt.scatter(x, y, color='blue', label='실제 데이터')
plt.plot(x, y_pred, color='red', label='회귀 직선')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()
# 오잉? 변수가 안좋네~

test = pd.read_csv("data/houseprice/test.csv")
x = np.array(test["OverallQual"]).reshape(-1, 1) # 이것도 바꿨어야됐네~
y_pred = model.predict(x)

sample_submission = pd.read_csv("data/houseprice/sample_submission.csv")
sample_submission["SalePrice"] = predict_y
sample_submission.to_csv("sample_submission_240802-7.csv", index = False)
# 이거는 6.64 나옴...
==========================================================================
# "GrLivArea" => "sample_submission_240802-6.csv" => 0.29117

==========================================================================
########## 이번엔 내가 변수 선택 2 ######### + 이상치 빼고 학습
train = pd.read_csv("data/houseprice/train.csv")

# 밑에 시각화로 중간점검 했을 때 이상치가 보인다...
train.query('GrLivArea > 4500')

# 이상치 2개 뺀거 데이터프레임으로 만들기
train = train.query('GrLivArea <= 4500')

### 회귀모델로 분석하기.

x = np.array(train["GrLivArea"]).reshape(-1, 1) # 마감 퀄리티로 해보자!
y = np.array(train["SalePrice"])

model = LinearRegression()

model.fit(x, y)

model.coef_         # 기울기 a
model.intercept_    # 절편 b
slope = model.coef_[0] # 어레이로 나와서 원소 뺴야됨
intercept = model.intercept_

y_pred = model.predict(x)

# 시각화로 중간 점검!
plt.scatter(x, y, color='blue', label='실제 데이터')
plt.plot(x, y_pred, color='red', label='회귀 직선')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

test = pd.read_csv("data/houseprice/test.csv")
x = np.array(test["GrLivArea"]).reshape(-1, 1)
y_pred = model.predict(x)

sample_submission = pd.read_csv("data/houseprice/sample_submission.csv")
sample_submission["SalePrice"] = y_pred
# sample_submission.to_csv("sample_submission_240802-9.csv", index = False)
# 0.29117 나이스나이스~~
==========================================================================


==========================================================================
###### 변수 2개 선택
train = pd.read_csv("data/houseprice/train.csv")
test = pd.read_csv("data/houseprice/test.csv")
sample_submission = pd.read_csv("data/houseprice/sample_submission.csv")

# 변수 선택
x = np.array(train[["GrLivArea", "GarageArea"]]).reshape(-1, 2)
# 변수 2개를 한다??? 그러면 두개 넣고 리쉐잎 하면 됨.
# 하지만 왠지 데이터프레임으로 하면 넣어질 것 같은데?
x = train[["GrLivArea", "GarageArea"]]
# 이게 되네;;
# 그러면 변수 하나만 넣을 때도 데이터프레임으로 하면 되지 않을까?
x = train[["GrLivArea"]]
# 이게 되네;;
# 데이터프레임으로 넣으면 어레이로 바꿀 필요도, 리쉐잎 할 필요도 없구나?

# 암튼
x = train[["GrLivArea", "GarageArea"]]
y = np.array(train["SalePrice"])

model = LinearRegression()

model.fit(x, y)

model.coef_         # 기울기 a
model.intercept_    # 절편 b
slope = model.coef_
intercept = model.intercept_

# f(x,y) = ax, by +c 함수 만들어보기
# def my_houseprice(x, y):
#     return slope[0] * x + slope[1] * y + intercept
#my_houseprice(train["GrLivArea"], train["GarageArea"])

y_pred = model.predict(x)

plt.scatter(x, y, color='blue', label='실제 데이터')
plt.plot(x, y_pred, color='red', label='회귀 직선')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

test_x = test[["GrLivArea", "GarageArea"]]
y_pred = model.predict(test_x) #에러 뜸.

# 결측치 확인.
test_x["GrLivArea"].isna().sum()
test_x["GarageArea"].isna().sum() #결측치 있음.
test_x = test_x.fillna(test["GarageArea"].mean())

y_pred = model.predict(test_x)

sample_submission["SalePrice"] = y_pred #셈플에 가격 넣기
#sample_submission.to_csv("sample_submission_240802-10.csv", index = False)
