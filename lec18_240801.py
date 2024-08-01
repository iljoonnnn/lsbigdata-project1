import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 직선 그려보기
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


### 집값 데이터 활용해서 선 그리기
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

test = pd.read_csv("data/houseprice/test.csv")
test = test.assign(SalePrice = ((80* test["BedroomAbvGr"] -30) * 1000))
test["SalePrice"]

sample_submission = pd.read_csv("data/houseprice/sample_submission.csv")
sample_submission["SalePrice"] = test["SalePrice"]
# sample_submission.to_csv("sample_submission_240801.csv", index = False)
