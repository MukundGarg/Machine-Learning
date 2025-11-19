import pandas as pd
from matplotlib import pyplot as plt
df = pd.read_csv("../../data/HR_comma_sep.csv")
# print(df)
# print(df.columns)
left = df[df.left==1]
left.shape
retained = df[df.left==0]
retained.shape
print (left)
print(retained)