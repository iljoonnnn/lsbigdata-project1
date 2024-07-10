#Ctrl + Enter
a=1
1+1
1+2
#Shift 누르면서 화살표 위 키 누르면 모두 선택되고 같이 실행할 수 있다.

#Show Folder in New Window: 해당위치 탐색기

#파워쉘 명령어어
# ls: 파일목록
# cd: 폴더이동
# .   현재폴더
# ..  상위폴더
# 상위 폴더로 가고 싶으면? "cd .."
# 두 번 가고 싶으면? "cd ..\.."
# \ : 앞 뒤 이어주는 기호
# Tab: 자동완성
  # Shift Tab: 다음 선택
# cls: 청소
#백스페이스 밑에 원 표시 누르면 \입력됨. 이어주는 명령어임.

a = 10
a = "안녕하세요!"
a
a = '안녕하세요!'
a
a = "'안녕하세요' 라고 아빠가 말했다."
a
a = '"안녕하세요" 라고 아빠가 말했다.'
a
a= [1,2,3]
var2 = [4,5,6]
a+var2

a='안녕하세요!'
b='LS 빅데이터 스쿨!'
a+b
a+' '+b #공백 넣기

print(a)
num1 = 3
num2 = 5
num1 + num2

#산술연산자자
a = 10
b = 3.3

print("a + b =", a + b)
print("a - b =", a - b)
print("a * b =", a * b)
print("a / b =", a / b)
print("a % b =", a % b)
print("a // b =", a // b)
print("a ** b =", a ** b)

(a**2) // 7
(a**2) &7
(a**2) &7
# Shift + Alt + 아래화살표: 아래로 복사사
# Shift + Alt + 아래화살표: 아래로 복사사

# Ctrl + Alt + 아래화살표: 커서 여러여러여러개

a == b
a!=b
a<b
a>b
a<=b
a>=b

a = (2 ** 4  + (12453 // 7)) % 8
b = ((9 ** 7) / 12) * (36452 % 253)
a < b

user_age = 14
is_adult = user_age >= 18
print("성인입니까?", is_adult)

a = "True"
b = TRUE
c = true
d = True

TRUE = 991022
b = TRUE
b

a = True
b = False

a and b
a or b
not a

# True: 1
# False: 0
True + True
True + False
False + False

# and 연산자자
True and False
True and True
False and False
False and True

True  * False
True  * True
False * False
False * True

# or 연산자
True or False
False or False

min(a + b, 1)

# not
(True - 1)
not(True + True)

a = 3
a += 10
a
a -= 4
a %= 3
a += 12
a **= 2
a /= 7

str1 = "hello!"
str1 + str1
str1 * 3

repeated_str = str1 * 3
print("Repeated string:", repeated_str)

# 정수: int(eger)
# 실수: float (double)

# 단항 연산자
x = 5
x
+x
-x
~x

#binary
bin(5)
~5
bin(-6)

#함수, 패키지
#pip install pydataset
import pydataset
pydataset.data()
df = pydataset.data("AirPassengers")
df

