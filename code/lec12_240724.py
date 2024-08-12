
# !pip install scipy
from scipy.stats import bernoulli
bernoulli.pmf(k, p)
bernoulli.cdf(k, p)
bernoulli.ppf(q, p)
bernoulli.rvs(p, size, random_state)

# 확률 질량 함수 (pmf)
# 확률변수가 갖는 값에 해당하는 확률을 저장하고 있는 함수.
# 
bernoulli.pmf(1, 3)

### 캐글 데이터 불러오기
import pandas as pd
train_df = pd.read_csv("data/houseprice/train.csv")
price_mean = train_df["SalePrice"].mean()

submission_df = pd.read_csv("data/houseprice/sample_submission.csv")
submission_df["SalePrice"] = price_mean #train데이터의 집 가격 평균으로 바꾼다.

# submission_df.to_csv("sample_submission.csv")
# 이렇게 하면 인덱스 값이 생긴 csv 파일이 생김김
submission_df.to_csv("data/houseprice/sample_submission.csv", index = False)
