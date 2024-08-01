# 숙제 검사
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
mydata = pd.read_csv("https://docs.google.com/spreadsheets/d/1RC8K0nzfpR3anLXpgtb8VDjEXtZ922N5N0LcSY5KMx8/gviz/tq?tqx=out:csv&sheet=Sheet2")
# URL 뒤에 gviz/tq?tqx=out:csv&sheet=Sheet2 이게 없으면 안된다.
mydata = mydata[["ID", "이름"]]
np.random.seed(20240730)
np.random.choice(mydata["이름"], 2, replace = False) #replace = False를 붙여야 중복되지 않음.

# 교제 9장 실습
# !pip install pyreadstat
import pyreadstat
raw_welfare=pd.read_spss("data/Koweps_hpwc14_2019/Koweps_hpwc14_2019_beta2.sav")
welfare = raw_welfare.copy()
welfare.shape

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
welfare.shape

### 깻잎 논쟁
# 이게 왜 논쟁이지? 복습 스터디 때 물어보자.
# 왜냐면 이건 마치 "같이 식사 할 때마다 3그릇 씩 먹는 애인 어떤지" 물어보는 것과 다르지 않다고 느껴졌기 떄문이다.
# 어떤 사람은 그런 애인이 좋을거고, 어떤 사람은 별로라고 할거다.
# 사람마다 다를거고 그러면 나와 다른 의견의 사람이 있으면 "너는 그렇구나" 이러고 말면 되는거 아닌가?
# 나와 다른 의견을 가진 사람을 설득시키려 하다보니까 논쟁이 된건가?
# 현주님한테 대답 들음. "서로 다르구나~" 하고 넘어갈 수 있는게 아니구나

### 남녀 월급 차이 알아보기기

# 성별 변수 알아보기기
welfare["sex"].dtypes
welfare["sex"].value_counts()
welfare["sex"].isna().sum()

# 원소 변경경
welfare["sex"] = np.where(welfare["sex"] == 1,
                          "male", "female")
welfare["sex"].value_counts()

# 월급 변수 알아보기기
welfare["income"].describe()
welfare["income"].value_counts()
welfare["income"].isna().sum()

sum(welfare["income"] > 9998)
welfare["income"].isna().sum()

sex_income = welfare.dropna(subset = "income") \
       .groupby("sex", as_index=False) \
       .agg(mean_income = ("income", "mean"))
sex_income

sns.barplot(data = sex_income, x = "sex", y = "mean_income", hue = "sex")
plt.show()
plt.clf()

# 숙제: 위 그래프에서 각 성별 95% 신뢰구간 계산 후 그리기
# 위 아래 검정색 막대기로 표시
# 현재 출력된게 평균이니까 평균 기준에서 대칭의 길이로 있겠지.

### 나이와 월급의 관계
welfare["birth"].describe() #연도가 float 이구만?
sns.histplot(data = welfare, x = "birth")
plt.show()
plt.clf()

sum(welfare["birth"]>9998) #무응답. 9999넘은 사람 있는지 확인
welfare["birth"].isna().sum()

welfare = welfare.assign(age = 2019 - welfare["birth"] + 1)
welfare["age"]
sns.histplot(data = welfare, x = "age")
plt.show()
plt.clf()

age_income = welfare.dropna(subset = "income") \
                    .groupby("age", as_index = False) \
                    .agg(age_mean = ("income", "mean"))
sns.lineplot(data = age_income, x = "age", y = "age_mean")
plt.show()
plt.clf()

# 월급이 0인 사람은 몇 명?
sum(welfare["income"] == 0)

# 월급 부분이 Nan인 사람은 몇 명? 무응답자 수 계산.
my_df = welfare.assign(income_na = welfare["income"].isna()) \
               .groupby("age", as_index = False) \
               .agg(n = ("income_na", "sum"))
my_df
sns.lineplot(data = my_df, x = "age", y = "n")
plt.show()
plt.clf()

### 연령대에 따른 월급 차이.
welfare = welfare.assign(ageg = np.where(welfare["age"] < 30, "young",
                                np.where(welfare["age"] <= 59, "middle",
                                                               "old")))
# 나이대 빈도 막대 그리기기
welfare["ageg"].value_counts()
sns.countplot(data = welfare, x = "ageg")
plt.show()
plt.clf()

# 나이대별 월급 평균 시각화
ageg_income = welfare.dropna(subset = "income") \
                     .groupby("ageg") \
                     .agg(mean_income = ("income", "mean"))

ageg_income
sns.barplot(data = ageg_income, x ="ageg", y = "mean_income", hue = "ageg",
                   order = ["young", "middle", "old"])
plt.show()
plt.clf()

# 10단위로 나이 잘라서 해보기.
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
welfare["agegg"]
agegg_income = welfare.dropna(subset = "income") \
                      .groupby("agegg") \
                      .agg(mean_income = ("income", "mean"))
sns.barplot(data = agegg_income, x ="agegg", y = "mean_income", hue = "agegg")
plt.show()
plt.clf()

# 10단위로 자를 때 cut 명령어 사용해보기.
bin_cut = np.array([0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119])
my_label = np.arange(0, 120, 10).astype(str) + "대"
#위에 리스트 자동화 하는 방법 생각해보기.
welfare = welfare.assign(age_group = pd.cut(welfare["age"],
                                    bins = bin_cut,
                                    labels = my_label))
welfare["age_group"]
age_income = welfare.dropna(subset = "income") \
                    .groupby("age_group", as_index = False) \
                    .agg(mean_income = ("income", "mean"))
age_income
sns.barplot(data = age_income, x ="age_group", y = "mean_income", hue = "age_group")
plt.show()
plt.clf()

# 9-5 연령대 및 성별 월급 차이
np.version.version
welfare["age_group"].info()
welfare["agegg"].info()
welfare["age_group"] = welfare["age_group"].astype("object") #카테고리 타입을 오브젝트로 바꿈꿈
sex_income = \
    welfare.dropna(subset = "income") \
           .groupby(["age_group", "sex"], as_index = False) \
           .agg(mean_income = ('income', 'mean'))


sns.barplot(data = sex_income, x = "age_group", y = "mean_income", hue = "sex")
plt.show()
plt.clf()


## 연령대별, 성별 상위 4% 수입 찾아보세요!
# quantile 함수 알아보기
# 사실은 lambda 함수도 같이 활용하는거네...
x = np.arange(10)
np.quantile(x, q= 0.94)

def my_f(vec):
    return vec.sum()

mytest = \
    welfare.dropna(subset = "income") \
           .groupby(["age_group", "sex"], as_index = False) \
           .agg(pct_boundary = ("income", lambda x : my_f(x)))
# def 함수 되는지 테스트... 이게 되네?


top_4pct = \
    welfare.dropna(subset = "income") \
           .groupby(["age_group", "sex"], as_index = False) \
           .agg(pct_boundary = ("income", lambda x : np.quantile(x, q=0.96)))

top_4pct
sns.barplot(data = top_4pct, x = "age_group", y = "4pct_boundary", hue = "sex")
plt.show()
plt.clf()

# 람다(lambda) 함수
# lambda 인자 : 표현식
# def 함수보다 쉽게 할 수 있다.
(lambda x, y: x + y)(10, 30)
