import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import folium
#!pip install folium

### 자꾸 터져서 11-2 먼저 하기.
geo_seoul = json.load(open("data/SIG_Seoul.geojson", encoding = "UTF-8"))
geo_seoul["features"] # 하나의 원소가 점임. 하나의 구 경계를 나타내는 점들임.
type(geo_seoul)
len(geo_seoul)
geo_seoul.keys()

len(geo_seoul["features"][0])
geo_seoul["features"][0].keys()
geo_seoul["features"][0]["type"]
geo_seoul["features"][0]["properties"] # 행정 구역 코드 출력
geo_seoul["features"][0]["geometry"] # 위도, 경도 좌표 출력

# 한번 좌표들만 빼서 스케터플랏 해볼까?
coordinate_list = geo_seoul["features"][0]["geometry"]["coordinates"]
len(coordinate_list)
#coordinate_list[0]
len(coordinate_list[0])
#coordinate_list[0][0]
len(coordinate_list[0][0])
coordinate_array = np.array(coordinate_list[0][0])
coordinate_array
x = coordinate_array[:, 0]
y = coordinate_array[:, 1]

plt.plot(x, y) # 사직동 경계 그림
plt.show()
plt.clf()

# 만약에 그림 그리는거의 용량을 줄이고 싶어!
plt.plot(x[::2], y[::2])
plt.show()
plt.clf()

geo_seoul["features"][1]["properties"]
geo_seoul["features"][2]["properties"] # 숫자를 다르게 하면 동내가 바뀜

# 입력한 숫자에 따라 지도 플랏나오게 하기.
def drowing_map(num):
    gu_name = geo_seoul["features"][num]["properties"]['SIG_KOR_NM']
    coordinate_list = geo_seoul["features"][num]["geometry"]["coordinates"]
    coordinate_array = np.array(coordinate_list[0][0])
    x = coordinate_array[:, 0]
    y = coordinate_array[:, 1]
    
    plt.rcParams.update({"font.family" : "Malgun Gothic"})
    plt.plot(x[::2], y[::2])
    plt.title(gu_name)
    plt.show()
    plt.clf()


drowing_map(5)
drowing_map(21)


#########################################################################

# 서울시 전체 지도 그리기. 연습

# 서울의 구 갯수
gu_num = len(geo_seoul["features"])
# 25

# 좌표 갯수
coordinate_list = geo_seoul["features"][0]["geometry"]["coordinates"]
coordinate_array = np.array(coordinate_list[0][0])
len(coordinate_array)
# 각 구의 좌표 갯수

# 구 이름 불러오기
geo_seoul["features"][0]["properties"]['SIG_KOR_NM']

# 좌표 불러오기
coordinate_list = geo_seoul["features"][0]["geometry"]["coordinates"]
coordinate_array = np.array(coordinate_list[0][0])
x = coordinate_array[:, 0]
y = coordinate_array[:, 1]
# 연습 끝

=======================================================================

# 리스트를 쌓는 반복문
name = []
x = []
y = []
gu_num = len(geo_seoul["features"])
for i in range(gu_num):
    coordinate_list = geo_seoul["features"][i]["geometry"]["coordinates"]
    coordinate_array = np.array(coordinate_list[0][0])
    gu_x = coordinate_array[:, 0]
    gu_y = coordinate_array[:, 1]
    x.extend(gu_x)
    y.extend(gu_y)
    coor_num = len(coordinate_array)
    for j in range(coor_num):
        gu_name = geo_seoul["features"][i]["properties"]['SIG_KOR_NM']
        name.append(gu_name)


len(name) #38231 # 총 좌표 갯수임.
len(x)
len(y)

# 리스트를 합쳐서 데이터프레임 만들기
mydata = pd.DataFrame({"name" : name,
                       "x" : x,
                       "y" : y})

# 시각화
sns.scatterplot(data = mydata, x = "x", y = "y", hue = "name", size = 0.005)
plt.show()
plt.clf()

=======================================================================

# 선생님 방법
def make_seouldf(num):
    gu_name = geo_seoul["features"][num]["properties"]['SIG_KOR_NM']
    coordinate_list = geo_seoul["features"][num]["geometry"]["coordinates"]
    coordinate_array = np.array(coordinate_list[0][0])
    x = coordinate_array[:, 0]
    y = coordinate_array[:, 1]
    return pd.DataFrame({"guname" : gu_name,
                         "x": x,
                         "y": y})

make_seouldf(1) # 맙소사...이게 되네?

result = pd.DataFrame({})

for i in range(25):
    result = pd.concat([result, make_seouldf(i)], ignore_index = True)

result

sns.scatterplot(data = result, x = "x", y = "y", hue = 'guname', s= 2, legend = False)
plt.show()
plt.clf()

# 강남만 따로 그려보기
gangnam_df = result.assign(is_gangnam = (np.where(result["guname"] == "강남구", "강남", "안강남")))
sns.scatterplot(
    data = gangnam_df,
    x = "x",
    y = "y",
    legend = False,
    hue = "is_gangnam",
    palette={"강남" : "grey", "안강남" : "red"},
    s = 2)

plt.show()
plt.clf()
# 팔레트에 색깔 설정 가능.


######################################################
######################################################
# 인구 데이터랑 합치기.
geo_seoul = json.load(open('data/SIG_Seoul.geojson', encoding = 'UTF-8'))
geo_seoul["features"][0]["properties"]

df_pop = pd.read_csv("data/Population_SIG.csv")
df_pop.head()
df_seoulpop = df_pop.iloc[1:26]
df_seoulpop["code"] = df_seoulpop["code"].astype(str)
df_seoulpop.info()

# import folium
# 흰 도화지 맵 가져오기
map_sig = folium.Map(location = [37.551, 126.973],
                    zoom_start = 12,
                    tiles = "cartodbpositron")
# 포로플릿
folium.Choropleth(
    geo_data = geo_seoul,
    data = df_seoulpop,
    columns = ("code", "pop"),
    key_on = "feature.properties.SIG_CD").add_to(map_sig)

map_sig.save("map_seoul.html")
# 산점도로 그렸던 그림을 맵 위에 덮어서 그린다는 뜻.
# 잘 안되는데 나중에 해보자.

# 구간을 나눠서 색깔이 확실하게 해보자.
bins = df_seoulpop["pop"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1])
folium.Choropleth(
    geo_data = geo_seoul,
    data = df_seoulpop,
    columns = ("code", "pop"),
    fill_color = "viridis",
    bins = bins,
    key_on = "feature.properties.SIG_CD").add_to(map_sig)

map_sig.save("map_seoul.html")

# 점 찍는 법
# make_seouldf(0).iloc[:,1:3].mean()
make_seouldf(1)
folium.Marker([37.583744, 126.983800], popup="종로구").add_to(map_sig)
map_sig.save("map_seoul.html")

######################################################
######################################################

# 하우스 프라이스로 지도 그려라.
train_location = pd.read_csv("data/houseprice/house_price(location).csv")

# 에임즈 동네 위치 찾기
train_location["Longitude"].mean() # 경도
train_location["Latitude"].mean() # 위도

# 흰 도화지 가지고 오기
map_sig = folium.Map(location = [42.03448223395904, -93.64289689856655],
                    zoom_start = 12,
                    tiles = "cartodbpositron")
map_sig.save("map_aims.html")

# 집 점 찍기.

for i in range(len(train_location)):
    x_point = train_location[["Latitude"]].iloc[i,0]
    x_float = float(x_point)
    y_point = train_location[["Longitude"]].iloc[i,0]
    y_float = float(y_point)
    folium.Marker([x_float, y_float]).add_to(map_sig)

map_sig.save("map_aims.html")

######################################################
######################################################
