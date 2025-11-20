import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
df = pd.read_csv("data/canada.csv")
plt.ylabel("per capita income")
plt.scatter(df["year"], df["per capita income (US$)"], color="red", marker='+')

reg=linear_model.LinearRegression()
p = reg.fit(df[["year"]], df["per capita income (US$)"])
plt.plot(df["year"], reg.predict(df[["year"]]), color='black')
plt.show()
