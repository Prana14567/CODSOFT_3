# iris_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target


model = RandomForestClassifier()
model.fit(X, y)

st.title("🌸 Iris Flower Classifier")
st.write("Enter the measurements below to predict the iris species.")


sepal_length = st.slider("Sepal Length (cm)", float(X['sepal length (cm)'].min()), float(X['sepal length (cm)'].max()), 5.1)
sepal_width  = st.slider("Sepal Width (cm)",  float(X['sepal width (cm)'].min()),  float(X['sepal width (cm)'].max()), 3.5)
petal_length = st.slider("Petal Length (cm)", float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()), 1.4)
petal_width  = st.slider("Petal Width (cm)",  float(X['petal width (cm)'].min()),  float(X['petal width (cm)'].max()), 0.2)

input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=X.columns)
prediction = model.predict(input_data)[0]
species = iris.target_names[prediction]

st.subheader(f"🌼 Predicted Species: `{species}`")
