
# !pip install plotly ## pip를 사용해 Plotly 설치

## 파이썬에서 Plotly 라이브러리 모듈 로딩
import plotly.graph_objects as go # 이게 정석
import plotly.express as px # 초보자용

import numpy as np
import pandas as pd
from palmerpenguins import load_penguins
from sklearn.linear_model import LinearRegression

# pip install nbformat

## 파이썬에서 Plotly 객체 초기화
df_covid19_100 = pd.read_csv("../data/df_covid19_100.csv")
df_covid19_100.info()

fig = go.Figure()
fig.show()

## Plotly 초기화 함수로 data 속성값 설정
fig = go.Figure(
    data = [
        {'type' : 'scatter',
        'mode' : 'markers',
        'x' : df_covid19_100.loc[df_covid19_100['iso_code'] == 'KOR', 'date'],
        'y' : df_covid19_100.loc[df_covid19_100['iso_code'] == 'KOR', 'new_cases'],
        'marker' : {'color' : 'red'}
        },
        {'type' : 'scatter',
        'mode' : 'lines',
        'x' : df_covid19_100.loc[df_covid19_100['iso_code'] == 'KOR', 'date'],
        'y' : df_covid19_100.loc[df_covid19_100['iso_code'] == 'KOR', 'new_cases'],
        'line' : {'color' : 'blue', 'dash' : 'dash'}
        }])

fig.show()

## 모드 lines로 변경.
fig = go.Figure(
    data = [
        {'type' : 'scatter',
        'mode' : 'lines',
        'x' : df_covid19_100.loc[df_covid19_100['iso_code'] == 'KOR', 'date'],
        'y' : df_covid19_100.loc[df_covid19_100['iso_code'] == 'KOR', 'new_cases'],
        'marker' : {'color' : 'red'}
        },
        {'type' : 'scatter',
        'mode' : 'lines',
        'x' : df_covid19_100.loc[df_covid19_100['iso_code'] == 'KOR', 'date'],
        'y' : df_covid19_100.loc[df_covid19_100['iso_code'] == 'KOR', 'new_cases'],
        'line' : {'color' : 'blue'}
        }])

fig.show()

## 레이아웃 속성
margins_P = {'t':50, 'b':25, 'l':25, 'r':25}

fig = go.Figure(
    data = {
        'type' : 'scatter',
        'mode' : 'markers+lines',
        'x' : df_covid19_100.loc[df_covid19_100['iso_code'] == 'KOR', 'date'],
        'y' : df_covid19_100.loc[df_covid19_100['iso_code'] == 'KOR', 'new_cases'],
        'marker' : {'color' : 'red'},
        'line' : {'color' : 'blue', 'dash' : 'dash'}
    },
    layout = {
        'title' : "코로나19 발생 현황",
        'xaxis' : {'title' : "날짜", 'showgrid' : False},
        'yaxis' : {'title' : "확진자수"},
        'margin' : margins_P
    })

fig.show()

##############################################################
==============================================================================

## 쳇 지피티한테 물어본 "애니메이션" 코드
# 맨 아래까지 드래그 하고 실행해야됨

frames = []
dates = df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "date"].unique()

for i in dates:
    frame_data = {
        "data": [
            {
                "type": "scatter",
                "mode": "markers",
                "x": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= i), "date"],
                "y": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= i), "new_cases"],
                "marker": {"color": "red"}
            },
            {
                "type": "scatter",
                "mode": "lines",
                "x": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= i), "date"],
                "y": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= i), "new_cases"],
                "line": {"color": "blue", "dash": "dash"}
            }
        ],
        "name": str(i)
    }

frames.append(frame_data)
frames
type(frames) # list
len(frames) # 1
type(frames[0]) # dict
frames[0]

# 애니메이션을 위한 x, y 축 범위 설정
x_range = ['2022-10-03', '2023-01-11']
y_range = [8900, 88172]

# 애니메이션을 위한 레이아웃 설정하는 코드
margins_P = {"l": 25, "r": 25, "t": 50, "b": 50}
layout = {
    "title": "코로나 19 발생현황",
    "xaxis": {"title": "날짜", "showgrid": False},
    "yaxis": {"title": "확진자수"},
    "margin": margins_P,
    "updatemenus": [{
        "type": "buttons",
        "showactive": False,
        "buttons": [{
            "label": "Play",
            "method": "animate",
            "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
        }, {
            "label": "Pause",
            "method": "animate",
            "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
        }]
    }]
}

# Figure 생성
fig = go.Figure(
    data=[
        {
            "type": "scatter",
            "mode": "markers",
            "x": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "date"],
            "y": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "new_cases"],
            "marker": {"color": "red"}
        },
        {
            "type": "scatter",
            "mode": "lines",
            "x": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "date"],
            "y": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "new_cases"],
            "line": {"color": "blue", "dash": "dash"}
        }
    ],
    layout=layout,
    frames=frames
)

fig.show()

==============================================================================

#######################################################################
#######################################################################

# 122 페이지

# 최근 뜨고있는 교육용 데이터 프레임 팽귄스.

## 데이터 불러오기
#!pip install palmerpenguins
from palmerpenguins import load_penguins
penguins = load_penguins()
penguins.head()
penguins["species"].unique()
penguins.columns

# 초기 산점도 생성
fig = px.scatter(
    penguins,
    x = "bill_length_mm",
    y = "bill_depth_mm",
    color = 'species')
fig.show()

# 점 크기 조정 (점 크기를 12로 설정)
fig.update_traces(marker=dict(size=12))

fig.update_layout(
    title=dict(
        text="팔머펭귄 종별 부리 길이 vs. 깊이",
        font=dict(size=20, color="white")
    ),
    paper_bgcolor= "black",
    plot_bgcolor= "black",
    font= dict(color="white"),
    xaxis= dict(
        title= dict(
            text="부리 길이 (mm)",
            font= dict(size = 16, color="white")
        ),
        tickfont= dict(size = 14, color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    yaxis= dict(
        title= dict(
            text="부리 깊이 (mm)",
            font= dict(size = 16, color="white")
        ), 
        tickfont= dict(size = 14, color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    legend= dict(font= dict(size = 14, color="white")),
)

fig.show()

#####################################################################

## 선형회귀 분석

from sklearn.linear_model import LinearRegression

model = LinearRegression()
penguins = penguins.dropna()
x = penguins[["bill_length_mm"]]
y = penguins["bill_depth_mm"]

model.fit(x, y)
linear_fit = model.predict(x) #x로 예측되는 y

fig.show() #이미 그린거

# 이미 그린거에 추가하기
fig.add_trace(
    go.Scatter(
        mode = "lines",
        x = penguins["bill_length_mm"],
        y = linear_fit,
        name = "선형회귀직선",
        line = dict(color = "white", dash = "dot")))
fig.show()

model.coef_ # 기울기 # 코이프라고 읽음
model.intercept_ # 절편
# 부리 길이가 증가할 때마다 부리 깊이가 줄어든다.

## 다른 방법
# 초기 산점도 생성할 때 할 수 있다.
fig = px.scatter(
    penguins,
    x = "bill_length_mm",
    y = "bill_depth_mm",
    color = 'species',
    trendline = "ols")
fig.show()

# 이렇게 다르게 나오는 걸 "심슨스페러독스" 라고 한다!!

##########################################################

## 범주형 변수로 회귀분석 진행하기
# 새로운 함수 등장!@! 두둥탁! pd.get_dummies
penguins_dummies = pd.get_dummies(
    penguins,
    columns = ["species"],
    drop_first=False)
# drop_first=False하면 3개가 다생김.

penguins_dummies
penguins_dummies.columns
penguins_dummies.iloc[:,-3:]
# 하지만 두개만 있어도 어떤 종인지 알 수 있기 때문에 2개만 있어도 됨.

penguins_dummies = pd.get_dummies(
    penguins,
    columns = ["species"],
    drop_first=True)
penguins_dummies
penguins_dummies.columns
penguins_dummies.iloc[:,-3:]

# x와 y설정
x = penguins_dummies[['bill_length_mm', 'species_Chinstrap', 'species_Gentoo']]
y = penguins_dummies['bill_depth_mm']

# 모델 학습
model = LinearRegression()
model.fit(x, y)

model.coef_ # 기울기 # 코이프라고 읽음
model.intercept_ # 절편

## 회귀직선 써보기
# y = 10.56 + 'bill_length_mm' * 0.2 + 'species_Chinstrap' * -1.9 + 'species_Gentoo' * -5.1
regline_y = model.predict(x)

=======================================================================

## 집값 시작
house_train = pd.read_csv("data/houseprice/train.csv")
house_test = pd.read_csv("data/houseprice/test.csv")
house_sample_submission = pd.read_csv("data/houseprice/sample_submission.csv")

## 회귀분석 적합(fit) 하기
train_x = house_train[["GrLivArea", "GarageArea", "Neighborhood"]] # 근데 이거 안 쓸거임.
train_y = np.array(house_train["SalePrice"])

train_x["Neighborhood"].unique()

house_dummies = pd.get_dummies(
    house_train,
    columns = ["Neighborhood"],
    drop_first=True)
# 기존 데이터에 더미코드 추가돼서 출력.
house_dummies

neighborhood_dummies = pd.get_dummies(
    house_train["Neighborhood"],
    drop_first=True)
# 이렇게 하면 또 다르다.
# 더미코드만 반환
neighborhood_dummies

train_x = pd.concat([
    house_train[["GrLivArea", "GarageArea"]], neighborhood_dummies
    ], axis=1)

## 회귀분석 모델링 하기!!!
house_train_model = LinearRegression()
house_train_model.fit(train_x, train_y)

house_train_model.coef_ # 기울기 # 코이프라고 읽음
house_train_model.intercept_ # 절편

# 모델링 한거 트레인 데이터로 예측해보기.
house_predict_for_train_x = house_train_model.predict(train_x)

## test 데이터에도 더미컬럼을 만들어준다.
neighborhood_dummies_test = pd.get_dummies(
    house_test["Neighborhood"],
    drop_first=True)

test_x = pd.concat([
    house_test[["GrLivArea", "GarageArea"]], neighborhood_dummies_test
    ], axis=1)

# 결측치 확인.
test_x.isna().sum() #결측치 있음.
test_x = test_x.fillna(house_test["GarageArea"].mean())

## 모델링 한거 실제로 예측.
house_predict_for_text_x = house_train_model.predict(test_x)

## 예측한 값 집어넣기
house_sample_submission["SalePrice"] = house_predict_for_text_x
# house_sample_submission.to_csv("data/houseprice/sample_submission_240808.csv", index = False)

