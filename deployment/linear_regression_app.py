import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("data/canada.csv")
st.write(df.head())  

plt.ylabel("per capita income")
plt.scatter(df["year"], df["per capita income (US$)"], color="red", marker='+')

reg = linear_model.LinearRegression()
p = reg.fit(df[["year"]], df["per capita income (US$)"])
plt.plot(df["year"], reg.predict(df[["year"]]), color='black')


st.pyplot(plt)


year = st.number_input("Enter year:", min_value=1900, max_value=2100, value=2025)

if st.button("Predict"):
    prediction = reg.predict([[year]])
    st.success(f"Predicted Per Capita Income for {year}: **${prediction[0]:.2f}**")