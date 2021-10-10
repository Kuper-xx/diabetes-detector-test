#Description: This program detects if someone has dibetes
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

st.write("""
        # Diabetes detection
        Detect if someone has diabetes!
""")
image = Image.open('header.png')
st.image(image, use_column_width=True)

df = pd.read_csv('diabetes.csv')
st.subheader('Data information:')
st.dataframe(df)
st.write(df.describe())
chart = st.bar_chart(df)
#Split the data into independent X and dependent Y variables
X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values
#Split the data set into 75% training and 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
#Get the feature input from the user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 0)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 23)
    insulin = st.sidebar.slider('insulin', 0, 846, 30)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.43, 0.3725)
    age = st.sidebar.slider('age', 20, 81, 20)

    user_data = {'pregnancies': pregnancies,
                 'glucose': glucose,
                 'blood_pressure': blood_pressure,
                 'skin_thickness': skin_thickness,
                 'insulin' : insulin,
                 'BMI' : BMI,
                 'DPF' : DPF,
                 'age' : age
            }
    features = pd.DataFrame(user_data, index = [0])
    return features
#Store the user input into a variable
user_input = get_user_input()
#Display
st.subheader('User Input:')
st.write(user_input)

#Create and train the model
RFC = RandomForestClassifier()
RFC.fit(X_train, Y_train)
#Show metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, RFC.predict(X_test)) * 100)+'%')
#Store the model predictions in a variable
prediction = RFC.predict(user_input)
#Display the classification
st.subheader('Classification: ')
st.write(prediction)
