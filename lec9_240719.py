import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# E[X]
np.arange(4) * np.array([1,2,2,1]) / 6

def through_coin(num, p):
    x = np.random.rand(num)
    return np.where(x < p, 1, 0)

through_coin(100, 0.5)
through_coin_df = pd.DataFrame({'counts' : through_coin(100, 0.5)})

plt.clf()
result = through_coin(10000000, 0.5)
through_coin_df = pd.DataFrame({'counts' : result})
sns.countplot(data = through_coin_df, x = 'counts')
plt.show()

#예제 넘파이배열 생성
data = np.random.rand(100000)
plt.clf()
plt.hist(data, bins = 1000, alpha = 0.7, color='blue')
# bins는 구간을 나누는 정도이다.
# alpha는 색 연한 정도
plt.title('Histogram of Nimpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

#이번엔 정규분포 그려보기
a = []
for i in range(10000):
    data = np.random.rand(5)
    data_mean = data.mean()
    a.append(float(data_mean))

plt.clf()
plt.hist(a, bins = 500, alpha = 0.7, color='blue')
plt.title('Histogram of Nimpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

## 반복문 쓰지 말고 넘파이 행렬로 구하기.
b = np.random.rand(50000).reshape(-1,5).mean(axis = 1)
b = np.random.rand(50000, 5).mean(axis = 1) #이렇게도 된다.

plt.clf()
plt.hist(b, bins = 30, alpha = 0.7, color='blue')
plt.title('Histogram of Nimpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

