import streamlit as st
import pickle
import numpy as np
def load_model():
    with open ('saved_steps','rb') as file:
        data=pickle.load(file)  
    return data
dataset=load_model()
regressor=dataset["regressor"]

st.title("WELCOME TO DIABETES PREDICTION SYSTEM")
st.write("""### We need some information !""")

Pregnancies=st.slider("Pregnancies",0,17,5)
Glucose=st.slider("Glucose",0,199,50)
BloodPressure=st.slider("BloodPressure",0,122,50)
SkinThickness=st.slider("SkinThickness",0,99,50)
Insulin=st.slider("Insulin",0,900,500)
BMI=st.slider("BMI",0,100,50)
DiabetesPedigreeFunction=st.slider("DiabetesPedigreeFunction",0.01,3.0,2.5)
Age=st.slider("Age",10,90,50)
ok=st.button("Submit")


if ok:
    X=np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    X=X.astype(float)
    prediction=regressor.predict(X)
    if prediction==1:
        st.subheader("You are diabetic")


    else :
        st.subheader("You are not diabetic")
        

        




