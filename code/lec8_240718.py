import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#두 개의 백터를 합쳐 행렬 생성
matrix = np.column_stack(
    (np.arange(1,5),
    np.arange(12,16)))
print("행렬:\n", matrix)

a = np.arange(0,10)
a
b = np.arange(10,20)
b
np.column_stack((a,b)) #괄호가 두 개 들어가야되네!
np.vstack((a,b))


matcolumn_stackmatrix = np.vstack(
    (np.arange(1,5),
    np.arange(12,16))) #세로로 붙이기
print("행렬:\n", matrix)

#빈 행렬
np.zeros(5)
np.zeros([5,4])

np.column_stack((np.zeros(5), np.zeros([5,4]))) #이것도 합쳐지구만!

#리쉐입
np.arange(1,5)\
.reshape([2,2])

np.arange(1,7)
np.arange(1,7).reshape(2,3)
np.arange(1,7).reshape(2,-1) #-1 하면 알아서 계산해준다.

#퀴즈. 0에서 99까지 수 중 랜덤하게 50개 숫자를 뽑아서
# 5 by 10 행렬 만드세요.
np.random.seed(2024)
a = np.random.randint(0,100,50)
a
a.reshape(5,-1)

np.arange(1,21).reshape(4,5)
np.arange(1,21).reshape(4,5, order = "F") #순서 설정정

#인덱싱
mat_a = np.arange(1,21).reshape(4,5, order = "F")
mat_a[0,0]
mat_a[1,1]
mat_a[2,3]
mat_a[0:2, 3] #이렇게도 가능함!
mat_a[1:3, 1:4]

#행자리, 열자리 비어있는 경우 전체 행, 또는 열 선택택
mat_a
mat_a[3, :]
mat_a[3,::2]

#짝수 행만 선택하려면?
mat_b = np.arange(1,101).reshape(20, -1)
mat_b
mat_b[1::2, :]

mat_b[[1,4,6,14], :]

# 1부터 10까지의 수에 2를 곱한 값으로 5행 2열의 행렬 생성
x = np.arange(1, 11).reshape((5, 2)) * 2
print("행렬 x:\n", x)

x[[True, True, False, False, True], 0] #논리연산자로 된다.

mat_b[:,1]
mat_b[:,[1]]
mat_b[:,1].reshape(-1,1)

#필터링
mat_b[:,1][mat_b[:,1]%7==0]

#사진은 행렬이다.
import numpy as np
import matplotlib.pyplot as plt

# 난수 생성하여 3x3 크기의 행렬 생성
np.random.seed(2024)
img1 = np.random.rand(3, 3)
print("이미지 행렬 img1:\n", img1)

# 행렬을 이미지로 표시
plt.imshow(img1, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()

plt.clf()
np.random.randint(0, 10, 20).reshape(4,-1) / 9 #이렇게 하면 원소의 값이 0~1사이 값을 가짐.
np.random.randint(0, 256, 20).reshape(4,-1) / 255
plt.imshow(img1, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()


import urllib.request
img_url = "https://bit.ly/3ErnM2Q"
urllib.request.urlretrieve(img_url, "jelly.png")

#!pip install imageio
import imageio
import numpy as np
# 이미지 읽기
jelly = imageio.imread("jelly.png")
print("이미지 클래스:", type(jelly))
print("이미지 차원:", jelly.shape)
print("이미지 첫 4x4 픽셀, 첫 번째 채널:\n", jelly[:4, :4, 0])
jelly.shape
jelly[:,:,0]

# 원본 이미지 표시
plt.imshow(jelly)
plt.axis('off')
plt.show()

# 배열 뒤집기 (전치)
t_jelly = np.transpose(jelly, (1, 0, 2))
# 뒤집힌 이미지 표시
plt.imshow(t_jelly)
plt.axis('off')
plt.show()

# 흑백으로 변환
bw_jelly = np.mean(jelly[:, :, :3], axis=2)
# 흑백 이미지 표시
plt.imshow(bw_jelly, cmap='gray')
plt.axis('off')
plt.show()

plt.imshow(bw_jelly)
plt.axis('off')
plt.show()

# 두 개의 2x3 행렬 생성
mat1 = np.arange(1, 7).reshape(2, 3)
mat2 = np.arange(7, 13).reshape(2, 3)
print("행렬 mat1:\n", mat1)
print("행렬 mat2:\n", mat2)
# 3차원 배열로 합치기
my_array = np.array([mat1, mat2])
my_array.shape #(2, 2, 3)
#(겹쳐있는 정도, 행, 렬렬)
print("3차원 배열 my_array:\n", my_array)
print("3차원 배열의 크기 (shape):", my_array.shape)

my_array[0, :, :]
my_array[:, :, :-1]
my_array[:, :, ::2]
my_array[:, 0, :]
my_array[0, 1, [1,2]]

mat_x = np.arange(1, 101).reshape(5,5,4)
mat_x
mat_y = np.arange(1, 101).reshape(10,5,2)
mat_y

my_array2 = np.array([my_array, my_array])
my_array2
my_array2.shape

a = np.array([1,2,3], [4,5,6])
a.sum()
a.sum(axis=0)

#가장 큰 수는?
mat_b.max()

#행별 가장 큰 수는?
mat_b.max(axis= 1)

#열별 가장 큰 수는?
mat_b.max(axis=0)

a=np.array([1,3,2,5])
a.sum()

mat_b.cumsum(axis= 1)
mat_b.cumprod(axis = 1)

mat_b.flatten()
mat_b.reshape(2,5,5)

d=np.array([1,2,3,4,5,6,7])
d.clip(2,4)

d.tolist()

#확률변수(균일확률변수) 만들기.
np.random.rand(1)

def myf():
    print("a")
myf() #함수 만들기

def X(num):
    return np.random.rand(num)

X(6)

# 베르누이 확률변수 모수:p 만들어보세요!
import numpy as np

def through_coin(num, p):
    x = np.random.rand(num)
    return np.where(x < p, 1, 0)

result = through_coin(100, 0.5) # 100번 던지기기
result.mean()
result = through_coin(100000000, 0.5) # 억 번 던지기기
result.mean()
#많이 던지면 이론 값에 가깝게 된다. "대수의 법칙"

#새로운 확률 변수
# 가질 수 있는 값: 0, 1, 2
# 20%, 50%, 30%
def three_side_dice():
    x = np.random.rand(1)
    return np.where(x < 0.2, 0, np.where(x<0.7, 1, 2))

three_side_dice()

#result = three_side_dice(100)
#mytable = pd.DataFrame({'value' : result})
#mytable.groupby('value').agg(value count = ('value', 'count))
#sns.countplot(data = mytable)
#plt.show()

## 이번엔 확률 조정할 수 있게 하기
p = np.array([0.2, 0.5, 0.3])
def three_side_dice(p):
    x = np.random.rand(1)
    return np.where(x < p[0], 0, np.where(x< (p[0]+p[1]), 1, 2))

three_side_dice(p)

# 리스트.cumsum() 버젼젼
p = np.array([0.2, 0.5, 0.3])
def three_side_dice(p):
    x = np.random.rand(1)
    p_cumsum = p.cumsum()
    return np.where(x < p_cumsum[0], 0, np.where(x< p_cumsum[1], 1, 2))

p = np.array([0.2, 0.5, 0.3])
def three_side_dice(p):
    x = np.random.rand(1)
    return np.where(x < p[0], '★', np.where(x< (p[0]+p[1]), '■', '♠'))


★ ■ ♠
three_side_dice(p)




