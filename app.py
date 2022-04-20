import streamlit as st
import pandas as pd
from PIL import Image
from matplotlib.pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np



data = pd.read_csv(r"C:\Users\HP\PycharmProjects\streamlit_learn\Salary_Data.csv")
x = np.array(data['YearsExperience']).reshape(-1,1)
lr = LinearRegression()
lr.fit(x,np.array(data['Salary']))

st.title("Salary Predictor")

image = Image.open(r"C:\Users\HP\PycharmProjects\streamlit_learn\sal.jpg")
# image = Image.open(r"C:\Users\HP\PycharmProjects\streamlit_learn\salary_image.jpg")
st.image(image, width=800)

nav = st.sidebar.radio("Navigation", ["Home", "Prediction", "Contribute"])

if nav == "Home":

    if st.checkbox("Show Table"):
        st.table(data=data)

    graph = st.selectbox("What kind of Graph ? ", ["Non-Interactive", "Interactive"])

    val = st.slider("Filter data using years", 0,20)
    data = data.loc[data["YearsExperience"] >=val]
    if graph == "Non-Interactive":
        fig, ax = plt.subplots()
        plt.figure(figsize=(10, 5))
        ax.scatter(data["YearsExperience"], data['Salary'])
        plt.ylim(0)
        plt.xlabel("Years of Experirence")
        plt.ylabel("Salary")
        plt.tight_layout()
        st.pyplot(fig)
    if graph == "Interactive":
        layout = go.Layout(
            xaxis=dict(range=[0, 16]),
            yaxis=dict(range=[0, 2100000])
        )
        fig = go.Figure(data=go.Scatter(x=data["YearsExperience"], y=data["Salary"], mode="markers"),
                        layout=layout)
        st.plotly_chart(fig)

if nav == "Prediction":
    st.header("Kmow your Salary")
    val = st.number_input("Enter yor exp",0.00,20.00,step=0.25)
    val = np.array(val).reshape(-1,1)
    pred = lr.predict(val)[0]

    if st.button("Predict"):
        st.success(f"Your predicted salary is {round(pred)}")

if nav == "Contribute":
    st.header("Cotribute to our dataset")
    ex = st.number_input("Enter your Experience",0.00,20.00)
    sal = st.number_input("Enter your Salary",0.00,1000000.00,step = 1000.00)
    if st.button("submit"):
            to_add = {"YearsExperience":[ex],"Salary":[sal]}
            to_add = pd.DataFrame(to_add)
            to_add.to_csv(r"C:\Users\HP\PycharmProjects\streamlit_learn\Salary_Data.csv",mode="a",header=False,index=False)
            st.success("Submitted")
