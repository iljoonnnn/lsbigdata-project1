import os
cwd = os.getcwd()
cwd # 현재 워킹디렉토리 알려주는 명령어

# 비쥬얼 스튜디오 주석처리: Ctrl + /

import pandas as pd
import numpy as np
import plotly.express as px
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

fig = px.scatter(
    penguins,
    x = "bill_length_mm",
    y = "bill_depth_mm",
    color = "species",
) # 저번에 했던 코드

fig.show()

fig.update_layout(
    title = {'text' : "팔머펭귄",
             'x': 0.5}
)
fig # vs코드는 그냥 fig 해도 출력 되구만.


fig.update_layout(
    title = {'text' : "팔머펭귄",
             'xanchor': "center"}
)
fig

## 70페이지 css
# <span> ... </span>
# 글자에 서식을 나타내는 구문
'''
<span>
    <span> ... </span>
    <span> ... </span>
    <span> ... </span>
</span>
'''

'''
<span>
    <span style = 'font-weight:bold'> ... </span> # 볼드처리
    <span> ... </span>
    <span> ... </span>
</span>
'''

fig.update_layout(
    title = {'text' : "<span style = 'font-weight:bold'> 팔머펭귄 </span>",
            }
)
fig
# 됐다!

## 글자 색상을 파란색으로 하고싶다면?
fig.update_layout(
    title = {'text' : "<span style = 'color:blue; font-weight:bold'> 팔머펭귄 </span>",
            }
)
fig


