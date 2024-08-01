import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ●●●●● 직선 그려보기
a = 1
b = 1
x = np.linspace(-5,5,100)
y = a*x + b
plt.plot(x, y)
plt.axvline(color = "black") # 원점 표시
plt.axhline(color = "black") # 원점 표시
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()
plt.clf()


# ●●●●● 집값 데이터 활용해서 선 그리기
train = pd.read_csv("data/houseprice/train.csv")

mydata = train[["BedroomAbvGr", "SalePrice"]].head(10)
mydata["SalePrice"] = mydata["SalePrice"]/1000
plt.scatter(data = mydata, x = "BedroomAbvGr", y = "SalePrice", color = "orange")
a = 80
b = -30
x = np.linspace(0,5,100)
y = a*x + b
plt.plot(x, y)
plt.show()
plt.clf()

# ●●●●● 그걸로 테스트 데이터에도 예측해서 캐글에 제출해보기
test = pd.read_csv("data/houseprice/test.csv")
test = test.assign(SalePrice = ((80* test["BedroomAbvGr"] -30) * 1000))
test["SalePrice"]

sample_submission = pd.read_csv("data/houseprice/sample_submission.csv")
sample_submission["SalePrice"] = test["SalePrice"]
# sample_submission.to_csv("sample_submission_240801.csv", index = False)


### 직선 성능 평가
a = 70
b = 10

# y_hat 어떻게 구할까?
y_hat = (a * train["BedroomAbvGr"] + b) * 1000

# y는 어디에 있는가?
y = train["SalePrice"]

np.abs(y - y_hat) # 절대값 해주는 명령어
np.sum(np.abs(y - y_hat)) # 다 더하기

### 한글 폰트 설정
plt.rcParams.update({"font.family" : "Malgun Gothic"})


# ●●●●● 선생님 코드
#!pip install scikit-learn

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 예시 데이터 (x와 y 벡터)
x = np.array([1, 3, 2, 1, 5]).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = np.array([1, 2, 3, 4, 5])  # y 벡터 (레이블 벡터는 1차원 배열입니다)

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_         # 기울기 a
model.intercept_    # 절편 b
slope = model.coef_[0] # 어레이로 나와서 원소 뺴야됨
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='실제 데이터')
plt.plot(x, y_pred, color='red', label='회귀 직선')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()


# ●●●●● 선생님이 준 코드 사용해서 [침실 수 - 집값] 회귀선 구하기.
train = pd.read_csv("data/houseprice/train.csv")
x = np.array(train["BedroomAbvGr"]).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = np.array(train["SalePrice"])  # y 벡터 (레이블 벡터는 1차원 배열입니다)

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_         # 기울기 a
model.intercept_    # 절편 b
slope = model.coef_[0] # 어레이로 나와서 원소 뺴야됨
intercept = model.intercept_
print(f"기울기 (slope): {slope}") # 16381.01698298878
print(f"절편 (intercept): {intercept}") # 133966.02049739176

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='실제 데이터')
plt.plot(x, y_pred, color='red', label='회귀 직선')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()


# ●●●●● 근데 방 8개인건 빼도 된다. 왜냐면 한개였던 것 같은데...
train.query('BedroomAbvGr == 8')

# 방 8개인거 로우 하나 뺀거 데이터프레임 만들기
iljoon2 = train.query('BedroomAbvGr != 8')[["BedroomAbvGr", "SalePrice"]]

x = np.array(iljoon2["BedroomAbvGr"]).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = np.array(iljoon2["SalePrice"])  # y 벡터 (레이블 벡터는 1차원 배열입니다)

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_         # 기울기 a
model.intercept_    # 절편 b
slope = model.coef_[0] # 어레이로 나와서 원소 뺴야됨
intercept = model.intercept_
print(f"기울기 (slope): {slope}") # 16734.59374102055
print(f"절편 (intercept): {intercept}") # 132998.31935829826

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='실제 데이터')
plt.plot(x, y_pred, color='red', label='회귀 직선')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

