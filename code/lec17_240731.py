from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import t

# 문자를 숫자로 변환
ord("a") #문자를 숫자로.
chr(97) #숫자를 문자로.
ord

# 숙제 검사 끝

# X ~ N(3, 7^2)
# 하위 25%에 해당하는 X 값은?
x = norm.ppf(0.25, loc = 3, scale = 7)

# Z ~ N(0, 1^2)
# 하위 25%에 해당하는 Z 값은?
z = norm.ppf(0.25, loc = 0, scale = 1)
z*7+3 # x랑 똑같다!!

# X ~ N(3, 7^2)
# 5 이하가 나올 확률은?
norm.cdf(5, loc = 3, scale = 7)

# Z의 세계에서는 "(5-3)/7" 이거겠지.
norm.cdf(2/7, loc=0, scale=1)

norm.ppf(0.975, loc=0, scale=1) # 1.96

### 표준정규분포에서 표본 1000개를 뽑아라. 히스토그램으로 그려라. -> pdf 겹쳐서 그리기
z = norm.rvs(loc=0, scale = 1, size = 1000)
z
sns.histplot(z, stat="density", color = "gray") #stat="density" 옵션을 쓰면 y축을 카운트에서 퍼센트로 바꿔준다.
plt.show()
zmin = z.min()
zmax = z.max()
z_values = np.linspace(zmin, zmax, 1000) #값 두개 넣은거 사이에 100개의 숫자를 만든다.
pdf_values = norm.pdf(z_values, loc=0, scale = 1) #각 값이 나올 수 있는 확률을 구한다. 그것은 나중에 높이가 된다.
plt.plot(z_values, pdf_values, color="red", linewidth=2)
plt.show()
plt.clf()

### X ~ N(3, 루트 2 ^2)
x = z*np.sqrt(2)+3 # 이론적인 식으로 덧뺄셈 하기
sns.histplot(x, stat="density", color = "green") #그걸로 히스트플랏
plt.show()
xmin = x.min()
xmax = x.max()
x_values = np.linspace(xmin, xmax, 1000) #값 두개 넣은거 사이에 100개의 숫자를 만든다.
pdf_values = norm.pdf(x_values, loc=3, scale = np.sqrt(2)) #각 값이 나올 수 있는 확률을 구한다. 그것은 나중에 높이가 된다.
plt.plot(x_values, pdf_values, color="blue", linewidth=2)
plt.show()
# 여기에 위에 그래프도 겹치기
z = norm.rvs(loc=0, scale = 1, size = 1000)
z
sns.histplot(z, stat="density", color = "gray") #stat="density" 옵션을 쓰면 y축을 카운트에서 퍼센트로 바꿔준다.
zmin = z.min()
zmax = z.max()
z_values = np.linspace(zmin, zmax, 1000) #값 두개 넣은거 사이에 100개의 숫자를 만든다.
pdf_values = norm.pdf(z_values, loc=0, scale = 1) #각 값이 나올 수 있는 확률을 구한다. 그것은 나중에 높이가 된다.
plt.plot(z_values, pdf_values, color="red", linewidth=2)
plt.show()
plt.clf()

# 정규분포 X ~ N(5, 3^2)
# (X-5) / 3 가 표준정규분포를 따르나요?
x = norm.rvs(loc=5, scale = 3, size = 1000)
x
z = (x-5) / 3 # 표준화임.
sns.histplot(z, stat="density", color = "gray") #stat="density" 옵션을 쓰면 y축을 카운트에서 퍼센트로 바꿔준다.
plt.show() #정규분포처럼 생기긴 했네... 하지만 "표준"정규분포일까?
zmin = z.min()
zmax = z.max()
z_values = np.linspace(zmin, zmax, 1000) #값 두개 넣은거 사이에 100개의 숫자를 만든다.
pdf_values = norm.pdf(z_values, loc=0, scale = 1) #각 값이 나올 수 있는 확률을 구한다. 그것은 나중에 높이가 된다.
plt.plot(z_values, pdf_values, color="red", linewidth=2)
plt.show()
plt.clf()

# 정규분포 X ~ N(5, 3^2)
# 1. X에서 표본 10개를 뽑아서 표본분산값 계산
# 2. X 표본 1000개 뽑음
# 3. 1번에서 계산한 S^2으로 시그마^2 대체한 표준화를 진행
# 표준화란? Z = (X-mu) / sigma
# 4. Z의 히스토그램 그리기

# 1.
np.random.seed(12345)
s_2 = norm.rvs(loc=5, scale = 3, size = 10000000).var()
# 2.
x = norm.rvs(loc=5, scale = 3, size = 1000)
# 3.
myarray = (x-5) / np.sqrt(s_2)
# 4.
sns.histplot(myarray, stat="density")
zmin = myarray.min()
zmax = myarray.max()
z_values = np.linspace(zmin, zmax, 1000) #값 두개 넣은거 사이에 1000개의 숫자를 만든다.
pdf_values = norm.pdf(z_values, loc=0, scale = 1) #각 값이 나올 수 있는 확률을 구한다. 그것은 나중에 높이가 된다.
plt.plot(z_values, pdf_values, color="red", linewidth=2)
plt.show()
plt.clf()

# 히스토그램 그려보니 둘이 맞을 때도 있고 아닐 때도 있다. 결국 이렇게 하는건 표준 정규분포로 바뀌지 않은것!

# 위에걸 20개로
s_2 = norm.rvs(loc=5, scale = 3, size = 20).var() # 20개!

x = norm.rvs(loc=5, scale = 3, size = 1000)

myarray = (x-5) / np.sqrt(s_2)

sns.histplot(myarray, stat="density")
zmin = myarray.min()
zmax = myarray.max()
z_values = np.linspace(zmin, zmax, 1000) #값 두개 넣은거 사이에 1000개의 숫자를 만든다.
pdf_values = norm.pdf(z_values, loc=0, scale = 1) #각 값이 나올 수 있는 확률을 구한다. 그것은 나중에 높이가 된다.
plt.plot(z_values, pdf_values, color="red", linewidth=2)
plt.show()
plt.clf
plt.show()
plt.clf()

# t 분포에 대해서 알아보자!
# X ~ t(df)
# 베르누이 분포처럼 모수를 하나 가짐.
# 하지만 연속함수이고 종모양이며 대칭이다. 중심은 0이다.
# 모수 df: 자유도라고 부름. 분산에 영향을 미친다. 퍼짐을 관장함.
from scipy.stats import t

# t.pdf
# t.ppf
# t.cdf
# t.rvs
?t.pdf
# 자유도가 4인 t분포의 pdf를 그려보세요!
t_values = np.linspace(-4, 4, 100) #값 두개 넣은거 사이에 100개의 숫자를 만든다.
pdf_values = t.pdf(t_values, df = 4) #각 값이 나올 수 있는 확률을 구한다. 그것은 나중에 높이가 된다.
plt.plot(t_values, pdf_values, color="red", linewidth=2)
plt.show()

# 표준준정규분포랑 비교해보자.
n_values = np.linspace(-4, 4, 100) #값 두개 넣은거 사이에 100개의 숫자를 만든다.
pdf_values = norm.pdf(n_values, loc=0, scale=1) #각 값이 나올 수 있는 확률을 구한다. 그것은 나중에 높이가 된다.
plt.plot(n_values, pdf_values, color="black", linewidth=2)
plt.show()
plt.clf()
# t분포에서 n이 무한대로 가면 표준정규분포처럼 된다.

# X ~ ?(mu, sigma^2/n)
# X bar ~ N(mu, sigma^2/n)
# X bar ~= t(x_bar, s^2/n) 자유도가 n-1인 t분포
x = norm.rvs(loc=15, scale= 3, size =16, random_state = 42)
x
n = len(x)

# t분포 이 때 졸음 ㅠ

### 교재 9-6

raw_welfare=pd.read_spss("data/Koweps_hpwc14_2019/Koweps_hpwc14_2019_beta2.sav")
welfare = raw_welfare.copy()

welfare = welfare.rename(
    columns = {
        "h14_g3" : "sex",
        "h14_g4" : "birth",
        "h14_g10" : "marriage_type",
        "h14_g11" : "religion",
        "p1402_8aq1" : "income",
        "h14_eco9" : "code_job",
        "h14_reg7" : "code_region"
    }
)

welfare = welfare[["sex", "birth", "marriage_type", "religion", "income", "code_job", "code_region"]]

# 원소 변경
welfare["sex"] = np.where(welfare["sex"] == 1,
                          "male", "female")
# 나이 변수 추가
welfare = welfare.assign(age = 2019 - welfare["birth"] + 1)

# 연령대 변수 추가가
welfare = welfare.assign(ageg = np.where(welfare["age"] < 30, "young",
                                np.where(welfare["age"] <= 59, "middle",
                                                               "old")))

# 연령대 10살 단위로로
welfare = welfare.assign(agegg = np.where(welfare["age"] < 10, "baby",
                                 np.where(welfare["age"] < 20, "10대",
                                 np.where(welfare["age"] < 30, "20대",
                                 np.where(welfare["age"] < 40, "30대",
                                 np.where(welfare["age"] < 50, "40대",
                                 np.where(welfare["age"] < 60, "50대",
                                 np.where(welfare["age"] < 70, "60대",
                                 np.where(welfare["age"] < 80, "70대",
                                 np.where(welfare["age"] < 90, "80대",
                                 np.where(welfare["age"] < 100, "90대",
                                 np.where(welfare["age"] < 110, "100대",
                                 np.where(welfare["age"] < 120, "110대", "120대"
                                )))))))))))))

top_4pct = \
    welfare.dropna(subset = "income") \
           .groupby(["agegg", "sex"], as_index = False) \
           .agg(pct_boundary = ("income", lambda x : np.quantile(x, q=0.96)))

top_4pct

top_1 = \
    welfare.dropna(subset = "income") \
           .groupby(["agegg", "sex"], as_index = False) \
           .agg(top_1_income = ("income", "max"))

top_1
#여기까지가 어제. lambda 함수를 쓰면 agg 기본 함수 외에도 여러가지 기능을 할 수 있다.

# 참고 남규형님꺼 코드(지원누님 아이디어어)     그룹바이를 이렇게도 쓸 수 있따.
welfare.dropna(subset = 'income') \
        .groupby('sex', as_index = False)[['income']] \
        .agg(['mean', 'std'])

welfare.dropna(subset = 'income') \
        .groupby('sex', as_index = False)[['income']] \
        .mean()
# agg를 굳이 안 써도 됨.

========================================

### 이제 진짜 오늘 진도
welfare["code_job"]
welfare["code_job"].value_counts()

list_job = pd.read_excel("data/Koweps_hpwc14_2019/Koweps_Codebook_2019.xlsx",
                         sheet_name = 1)
list_job.head()

# 데이터프레임 합치기
welfare = welfare.merge(list_job, how = "left", on="code_job")

# 월급 없는거 빼기
welfare.dropna(subset = ["job", "income"])[["job", "income"]]

# 맑은 고딕 폰트 설정
plt.rcParams.update({"font.family" : "Malgun Gothic"})

# 직업별 월급 평균표 만들기
job_income_top10 = welfare.dropna(subset = ["job", "income"]) \
                    .groupby("job", as_index = False) \
                    .agg(mean_income = ("income", "mean")) \
                    .sort_values("mean_income", ascending = False) \
                    .head(10)

# 시각화
sns.barplot(data = job_income_top10, x = 'mean_income', y = 'job', hue = "job")
plt.show()
plt.clf()

### 9-7 챕터
# 쿼리도 할 수 있다.
# 남자 성별 중 돈 가장 많이 버는 직업 top 10
job_income_top10_male = \
    welfare.dropna(subset = ["job", "income"]) \
           .query('sex == "male"') \
           .groupby("job", as_index = False) \
           .agg(mean_income = ("income", "mean")) \
           .sort_values("mean_income", ascending = False) \
           .head(10)
job_income_top10_male
sns.barplot(data = job_income_top10_male, x = 'mean_income', y = 'job', hue = "job")
plt.show()
plt.clf()

# 여성 성별 중 돈 가장 많이 버는 직업 top 10
job_income_top10_female = \
    welfare.dropna(subset = ["job", "income"]) \
           .query('sex == "female"') \
           .groupby("job", as_index = False) \
           .agg(mean_income = ("income", "mean")) \
           .sort_values("mean_income", ascending = False) \
           .head(10)
job_income_top10_female
sns.barplot(data = job_income_top10_female, x = 'mean_income', y = 'job', hue = "job")
plt.show()
plt.clf()

# 9-8_ p.263
df = \
    welfare.query('marriage_type != 5') \
           .groupby("religion", as_index = False) \
           ["marriage_type"] \
           .value_counts(normalize = True)

df = df.query('marriage_type == 3') \
       .assign(proportion = df["proportion"] * 100) \
       .round(1)

#노멀라이즈를 True로 하면 카운트 대신 비율로 나온다.
df


==========================================================


##### 집값 데이터를 이용한 여러가지 시각화화 ######

train = pd.read_csv("data/houseprice/train.csv")
train

#### 변수 선택 1.
# BedroomAbvGr : 지하실 이상의 침실 수
# OverallQual : 전체 재료 및 마감 품질
iljoon_data1 = train[["BedroomAbvGr", "OverallQual"]]
iljoon_data1

## 선택한 변수의 기본적인 시각화

# 방 갯수마다 집 갯수
sns.countplot(data = iljoon_data1, x = "BedroomAbvGr")
plt.show()
plt.clf()
# 방 3개인 집이 제일 많다.
# 최소 0개, 최대 8개이다.

# 마감 퀄리티별별 집 갯수
sns.countplot(data = iljoon_data1, x = "OverallQual")
plt.show()
plt.clf()
# 1부터 10등급까지 있다.
# 평균과 평균 이상인 집들이 가장 많다.

## 재료 및 마감 품질에 따른 침실 갯수는 어떨까?
sns.scatterplot(data = iljoon_data1, x = "OverallQual", y = "BedroomAbvGr")
plt.show()
plt.clf()
# 원하는 느낌이 아닌디...

### 빈도 수로 나타내보자!

# 퀄리티 이름 추가.
iljoon_data1 = iljoon_data1.assign(Qual_grade = (np.where(iljoon_data1["OverallQual"] == 1, "Very Poor",
                                                 np.where(iljoon_data1["OverallQual"] == 2, "Poor",
                                                 np.where(iljoon_data1["OverallQual"] == 3, "Fair",
                                                 np.where(iljoon_data1["OverallQual"] == 4, "Below Average",
                                                 np.where(iljoon_data1["OverallQual"] == 5, "Average",
                                                 np.where(iljoon_data1["OverallQual"] == 6, "Above Average",
                                                 np.where(iljoon_data1["OverallQual"] == 7, "Good",
                                                 np.where(iljoon_data1["OverallQual"] == 8, "Very Good",
                                                 np.where(iljoon_data1["OverallQual"] == 9, "Excellent", "Very Excellent",
                                                 )))))))))))
iljoon_data1

### 퀄리티 별 마다 방 갯수가 각각 몇개있는지.

# 1. Very Poor : 마감 상태 안 좋음.
count_Very_Poor = \
    iljoon_data1.query('Qual_grade == "Very Poor"')\
                .groupby("BedroomAbvGr", as_index = False) \
                .agg(Bedroom_count = ("BedroomAbvGr", "count"))
count_Very_Poor
sns.barplot(data = count_Very_Poor, x = "BedroomAbvGr", y = "Bedroom_count")
plt.title("1. Very Poor") #표에 이름 넣어줌.
plt.show()
plt.clf()

# 2. Poor
count_Poor = \
    iljoon_data1.query('Qual_grade == "Poor"')\
                .groupby("BedroomAbvGr", as_index = False) \
                .agg(Bedroom_count = ("BedroomAbvGr", "count"))
sns.barplot(data = count_Poor, x = "BedroomAbvGr", y = "Bedroom_count")
plt.title("2. Poor")
plt.show()
plt.clf()

# 3. Fair
count_Fair = \
    iljoon_data1.query('Qual_grade == "Fair"')\
                .groupby("BedroomAbvGr", as_index = False) \
                .agg(Bedroom_count = ("BedroomAbvGr", "count"))
sns.barplot(data = count_Fair, x = "BedroomAbvGr", y = "Bedroom_count")
plt.title("3. Fair")
plt.show()
plt.clf()

# 4. Below Average
count_Below_Average = \
    iljoon_data1.query('Qual_grade == "Below Average"')\
                .groupby("BedroomAbvGr", as_index = False) \
                .agg(Bedroom_count = ("BedroomAbvGr", "count"))
sns.barplot(data = count_Below_Average, x = "BedroomAbvGr", y = "Bedroom_count")
plt.title("4. Below Average")
plt.show()
plt.clf()

# 5. Average
count_Average = \
    iljoon_data1.query('Qual_grade == "Average"')\
                .groupby("BedroomAbvGr", as_index = False) \
                .agg(Bedroom_count = ("BedroomAbvGr", "count"))
sns.barplot(data = count_Average, x = "BedroomAbvGr", y = "Bedroom_count")
plt.title("5. Average")
plt.show()
plt.clf()

# 6. Above Average
count_Above_Average = \
    iljoon_data1.query('Qual_grade == "Above Average"')\
                .groupby("BedroomAbvGr", as_index = False) \
                .agg(Bedroom_count = ("BedroomAbvGr", "count"))
sns.barplot(data = count_Above_Average, x = "BedroomAbvGr", y = "Bedroom_count")
plt.title("6. Above Average")
plt.show()
plt.clf()

# 7. Good
count_Good = \
    iljoon_data1.query('Qual_grade == "Good"')\
                .groupby("BedroomAbvGr", as_index = False) \
                .agg(Bedroom_count = ("BedroomAbvGr", "count"))
sns.barplot(data = count_Good, x = "BedroomAbvGr", y = "Bedroom_count")
plt.title("7. Good")
plt.show()
plt.clf()

# 8. Very Good
count_Very_Good = \
    iljoon_data1.query('Qual_grade == "Very Good"')\
                .groupby("BedroomAbvGr", as_index = False) \
                .agg(Bedroom_count = ("BedroomAbvGr", "count"))
sns.barplot(data = count_Very_Good, x = "BedroomAbvGr", y = "Bedroom_count")
plt.title("8. Very Good")
plt.show()
plt.clf()

# 9. Excellent
count_Excellent = \
    iljoon_data1.query('Qual_grade == "Excellent"')\
                .groupby("BedroomAbvGr", as_index = False) \
                .agg(Bedroom_count = ("BedroomAbvGr", "count"))
sns.barplot(data = count_Excellent, x = "BedroomAbvGr", y = "Bedroom_count")
plt.title("9. Excellent")
plt.show()
plt.clf()

# 10. Very Excellent
count_Very_Excellent = \
    iljoon_data1.query('Qual_grade == "Very Excellent"')\
                .groupby("BedroomAbvGr", as_index = False) \
                .agg(Bedroom_count = ("BedroomAbvGr", "count"))
sns.barplot(data = count_Very_Excellent, x = "BedroomAbvGr", y = "Bedroom_count")
plt.title("10. Very Excellent")
plt.show()
plt.clf()

# 11. 한눈에 보기.
all_mean = \
    iljoon_data1.groupby("Qual_grade") \
                .agg(Bedroom_mean = ("BedroomAbvGr", "mean"))

sns.barplot(data = all_mean,
            x = "Bedroom_mean",
            y = "Qual_grade",
            hue = "Qual_grade",
            order = ["Very Poor", "Poor", "Fair", "Below Average", "Average", "Above Average", "Good", "Very Good", "Excellent", "Very Excellent"])
plt.title("all")
plt.show()
plt.clf()
# 마감상태가 안 좋으면 방이 하나라고 생각해도 된다.
# 그 외에는 방 갯수 2~3개 생각하고 가면 된다.







# Neighborhood : 아이오와주 에임스 시 내의 물리적 위치
# SalePrice : 매매 가격 (달러)



