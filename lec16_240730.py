# 숙제 검사
import pandas as pd
import numpy as np
mydata = pd.read_csv("https://docs.google.com/spreadsheets/d/1RC8K0nzfpR3anLXpgtb8VDjEXtZ922N5N0LcSY5KMx8/gviz/tq?tqx=out:csv&sheet=Sheet2")
mydata = mydata[["ID", "이름"]]
np.random.seed(20240730)
np.random.choice(mydata["이름"], 2, replace = False) #replace = False를 붙여야 중복되지 않음.

