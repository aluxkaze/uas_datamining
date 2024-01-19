import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the saved model
with open('iris.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the dataset for reference
iris_df = pd.read_csv('IRIS.csv')

# Create a simple form for user input
st.title('Iris Flower Prediction App')
st.write('Enter the values for Sepal Length, Sepal Width, Petal Length, and Petal Width to predict the Iris species')

# Add input fields for user to input data
sepal_length = st.number_input('sepal_length', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
sepal_width = st.number_input('sepal_width', min_value=0.0, max_value=10.0, value=3.0, step=0.1)
petal_length = st.number_input('petal_wength', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
petal_width = st.number_input('petal_width', min_value=0.0, max_value=10.0, value=0.1, step=0.1)

st.write('NIM: 2020230022')
st.write('Name: Satria Afit Hernando')

# Create a button to trigger the prediction
if st.button('Predict'):
    # Perform prediction using the loaded model
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)

    # Display the prediction result
    st.write('Predicted Iris Species:', iris_df['species'].unique()[prediction][0])
