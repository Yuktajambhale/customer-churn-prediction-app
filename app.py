# Gender -> 1 female  0 male
# Churn -> 1 yes 0 no   
# Scalar is exported as scaler.pkl
# model is exported as model.pkl    
# order of the X->'Age','Gender','Tenure','MonthlyCharges'

import streamlit as st
import joblib 
import numpy as np

scaler=joblib.load("scaler.pkl")
model=joblib.load("model.pkl")

st.title("Customer Churn Prediction")

st.divider()

st.write("Please provide the following details to predict whether the customer will churn or not.")

st.divider()

age = st.number_input("Enter Age", min_value=10, max_value=100, value=30)

tenure = st.number_input("Enter Tenure", min_value=0, max_value=130,value=10)

monthlycharge=st.number_input("Enter Monthly Charges", min_value=30,max_value=150)
    
gender=st.selectbox("Enter Gender",["Male","Female"])

st.divider()

predictbutton=st.button("Predict")  

if predictbutton:

    gender_selected = 1 if gender =="Female" else 0

    X=[age,gender_selected,tenure,monthlycharge]

    X1=np.array(X)

    X_array = scaler.transform([X1])


    prediction=model.predict(X_array)[0]

    predicted ="Churn" if prediction==1 else "Not Churn"

    st.balloons()

    st.write(f"Predicted: {predicted}")

else:
    st.write("please enter the values and the predict button")

