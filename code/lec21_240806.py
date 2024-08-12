import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
import seaborn as sns
import json
import folium
import matplotlib.pyplot as plt

# t검정 연습
tab3 = pd.read_csv("data/tab3.csv")
tab3

tab1 = tab3[["id", "score"]]
tab1["id"] = np.arange(1,13)
tab1

tab2 = tab1.assign(gender = (["female"] * 7 + ["male"] * 5)) #리스트의 성질
tab2

#######################################################################
# alternative = "two-sided" or greater or less

# 1
# 1 표본 t 검정 (그룹 1개)
# 귀무가설 vs 대립가설
# H0: mu = 10 vs Ha: mu != 10
# 유의수준 5%로 설정
from scipy.stats import ttest_1samp

result = ttest_1samp(tab1["score"], popmean = 10, alternative = "two-sided") #뮤가 10과 같지 않다 이기 때문에. 투사이드
type(result)
result[0]
result[1]
result[2] #없음
tab1["score"].mean() # 11.53
result.statistic # t 검정 통계량
result.pvalue # 유의확률 # 0.0648
result.df
ci = result.confidence_interval(confidence_level = 0.95) # 95% 유의수준에 대응하는 구간을 알 수 있다.
ci[0]
ci[1]
# 유의확률 0.0648이 우의수준 0.05보다 크므로
# 귀무가설을 기각하지 못한다.

# 좀 더 쉽게 풀기...
# 귀무가설이 참일 때(mu = 10), 11.53이 관찰될 확률: 6.48%
# 이것은 보기 힘들다고 판단하는 기준 0.05(유의수준)보다 크므로
# 평균이 10일 때도 관측될 수 있는 숫자라고 판단.
# 귀무가설 채택

# 그러면 0.05라고 한 순간. 귀무가설이 채택되는 범위가 결정 된거네?
# 11.53은 그 범위에 들어온거고.
# 바깥으로 빠지는 범위는 기각역이락 하구나. 유의수준은 바깥으로 빠지는 확률이구나.
#   그러면 궁금한거!! 5%의 유의수준이라고 정하면,
#   95%의 신뢰구간에 들어오면 mu=10 이라는 말이 맞는거네?
# 위에 말 다 맞는 것으로 확인 됨.

#######################################################################

# 2
# 분산 같은 경우: 독립 2표본 t검정
# 분산 다를 경우: 웰치스 t 검정

# 2표본 t 검정 (그룹 2) - 분산 같고, 다를 때
## 귀무가설 vs. 대립가설
## H0: mu_m = mu_f vs. Ha: mu_m > mu_f
## 유의수준 1%로 설정, 두 그룹 분산 같다고 가정한다.
female = tab2[tab2['gender'] == "female"] ['score']
male = tab2[tab2['gender'] == "male"] ['score']

from scipy.stats import ttest_ind

# alternative = "less" 의미는 대립가설이
# 첫 번째 입력그룹의 평균이 두 번째 입력 그룹 평균보다 작다.
# 고 설정된 경우를 나타냄.
# 대립가설 기준.
# equal_var = True 의미는 두 그룹의 분산이 같다는 가정.
# 분산이 같지 않을경우 False로 놓으면 됨.
result = ttest_ind(female, male, alternative = "less", equal_var = True)
result.statistic
result.pvalue

########################################################################

# 3
# 대응표본 t 검정 (짝지을 수 있는 표본)
## 귀무가설 vs. 대립가설
## H0: mu_before = mu_after vs. Ha: mu_after > mu_before
## H0: mu_d = 0 vs. Ha: mu_d > 0
## mu_d = mu_after - mu_before
## 유의수준 1%로 설정, 두 그룹 분산 같다고 가정한다.
tab3_data = tab3.pivot_table(index ="id",
                             columns ="group",
                             values ="score").reset_index()
tab3_data
# 피벗 테이블. id 기준으로 행 갯수가 정해지고 컬럼에 입력한 컬럼의 유니크한 원소들이 컬럼명으로 바뀐다. 채워지는건 value에 입력한 값.
tab3_data['score_diff'] = tab3_data['after'] - tab3_data['before']
tab3_data = tab3_data['score_diff']
tab3_data
result = ttest_1samp(tab3_data, popmean = 0, alternative = "greater")
result.statistic
result.pvalue
tab3_data.sum()
# 결론: 0보다 크긴 하지만 이렇게 나온 값이 0보다 크지 않을 확률이 2.4%임.
# 그것이 1%보다 높으므로 귀무가설 채택. 귀무가설을 기각하지 못한다.

# melt. 다시 길게 뽑기 연습
df = pd.DataFrame({"id" : [1, 2, 3],
                   "A" : [10, 20, 30],
                        "B" : [40, 50, 60]})
df

df_long = df.melt(id_vars = "id",
                  value_vars = ["A","B"],
                  var_name = "group",
                  value_name = "score")
df_long

df_long.pivot_table(
    columns = "group",
    values = "score"
).reset_index()

df_long.pivot_table(
    index = "id",
    columns = "group",
    values = "score"
).reset_index()

df_long.pivot_table(
    columns = "group",
    values = "score",
    aggfunc = "mean"
)
# 최댓값 하고 싶으면 aggfunc = "max"

# 연습 2
tips = sns.load_dataset("tips")

tips.pivot_table(
    columns = "day",
    values = "tip")

################################################################

# 교제 11장
import json
geo = json.load(open('data/SIG.geojson', encoding = 'UTF-8'))

# 행정 구역 코드 출력
geo['features'][0]['properties']

# 위도, 경도 좌표 출력
geo['features'][0]['geometry']

# 시군구 인구 데이터 준비하기
df_pop = pd.read_csv("data/Population_SIG.csv")
df_pop.head()
df_pop.info()
df_pop['code'] = df_pop['code'].astype(str)

# !pip install folium

import folium
folium.Map(location = [35.95, 127.7],
           zoom_start = 8)
plt.show()
plt.clf()









