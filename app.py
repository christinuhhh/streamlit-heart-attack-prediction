import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Define the feature list in the same order as used in training.
features = [
    'Age', 'Sex', 'Cholesterol', 'Heart Rate', 'Diabetes', 'Family History',
    'Smoking', 'Obesity', 'Alcohol Consumption', 'Exercise Hours Per Week', 'Diet',
    'Previous Heart Problems', 'Medication Use', 'Stress Level', 'Sedentary Hours Per Day',
    'Income', 'BMI', 'Triglycerides', 'Physical Activity Days Per Week', 'Sleep Hours Per Day',
    'Systolic', 'Diastolic'
]

# ---------------------------
# Functions to Load Models, Scalers, and Data
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_models_and_scalers(model_choice):
    """
    Load the selected model and shared scalers.
    """
    scalers_path = os.path.join("scalers", "scalers.pkl")
    scalers = joblib.load(scalers_path)
    
    if model_choice == "Logistic Regression":
        model_path = os.path.join("models", "logistic_regression_model.pkl")
    elif model_choice == "Random Forest":
        model_path = os.path.join("models", "random_forest_model.pkl")
    else:
        st.error("Invalid model choice!")
        st.stop()
        
    model = joblib.load(model_path)
    return model, scalers

@st.cache_data(show_spinner=False)
def load_preprocessed_data():
    """
    Load the preprocessed CSV data.
    """
    csv_path = "heart_attack_prediction_dataset_preprocessed.csv"
    data = pd.read_csv(csv_path)
    return data

# ---------------------------
# Prediction Function
# ---------------------------
def predict_diagnosis(datapoint: dict, model, scalers):
    """
    Given a datapoint (a dictionary of features), scale the input values and make a prediction.
    """
    datapoint_list = []
    for col in features:
        # Get the value; if missing, throw an error.
        if col in datapoint:
            value = datapoint[col]
            # Scale the value using the corresponding scaler.
            scaled_value = scalers[col].transform(np.array([[value]])).flatten()[0]
            datapoint_list.append(scaled_value)
        else:
            st.error(f"Missing value for column: {col}")
            st.stop()
    datapoint_array = np.array(datapoint_list).reshape(1, -1)
    prediction = model.predict(datapoint_array)[0]
    return prediction

# ---------------------------
# Streamlit Pages
# ---------------------------
def prediction_page():
    st.title("Heart Attack Risk Prediction")
    st.write("Enter the patient details below and choose a model for prediction.")

    # Let user choose which model to use
    model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest"])
    
    # Load the chosen model and scalers (cached)
    model, scalers = load_models_and_scalers(model_choice)
    
    # Create a form for input features.
    with st.form("prediction_form"):
        # Numeric inputs and some categorical selections.
        Age = st.number_input("Age", min_value=0, max_value=120, value=50, step=1)
        Sex = st.selectbox("Sex", options=["Male", "Female"])
        Sex = 1 if Sex == "Male" else 0
        Cholesterol = st.number_input("Cholesterol", min_value=0.0, value=200.0)
        Heart_Rate = st.number_input("Heart Rate", min_value=0, value=70, step=1)
        
        Diabetes = st.selectbox("Diabetes", options=["No", "Yes"])
        Diabetes = 1 if Diabetes == "Yes" else 0
        
        Family_History = st.selectbox("Family History", options=["No", "Yes"])
        Family_History = 1 if Family_History == "Yes" else 0
        
        Smoking = st.selectbox("Smoking", options=["No", "Yes"])
        Smoking = 1 if Smoking == "Yes" else 0
        
        Obesity = st.selectbox("Obesity", options=["No", "Yes"])
        Obesity = 1 if Obesity == "Yes" else 0
        
        Alcohol_Consumption = st.number_input("Alcohol Consumption (units per week)", min_value=0.0, value=0.0, step=0.5)
        Exercise_Hours = st.number_input("Exercise Hours Per Week", min_value=0.0, value=3.0, step=0.5)
        
        Diet = st.selectbox("Diet", options=["Unhealthy", "Average", "Healthy"])
        if Diet == "Unhealthy":
            Diet = 0
        elif Diet == "Average":
            Diet = 1
        else:
            Diet = 2
        
        Previous_Heart_Problems = st.selectbox("Previous Heart Problems", options=["No", "Yes"])
        Previous_Heart_Problems = 1 if Previous_Heart_Problems == "Yes" else 0
        
        Medication_Use = st.selectbox("Medication Use", options=["No", "Yes"])
        Medication_Use = 1 if Medication_Use == "Yes" else 0
        
        Stress_Level = st.number_input("Stress Level (0-10)", min_value=0, max_value=10, value=5, step=1)
        Sedentary_Hours = st.number_input("Sedentary Hours Per Day", min_value=0.0, value=8.0, step=0.5)
        Income = st.number_input("Income", min_value=0.0, value=50000.0, step=1000.0)
        BMI = st.number_input("BMI", min_value=0.0, value=24.0, step=0.1)
        Triglycerides = st.number_input("Triglycerides", min_value=0.0, value=150.0, step=1.0)
        Physical_Activity_Days = st.number_input("Physical Activity Days Per Week", min_value=0, max_value=7, value=3, step=1)
        Sleep_Hours = st.number_input("Sleep Hours Per Day", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        Systolic = st.number_input("Systolic", min_value=0, max_value=300, value=120, step=1)
        Diastolic = st.number_input("Diastolic", min_value=0, max_value=200, value=80, step=1)

        submitted = st.form_submit_button("Predict")
    
    if submitted:
        # Assemble all input values into a dictionary matching the feature list.
        datapoint = {
            'Age': Age,
            'Sex': Sex,
            'Cholesterol': Cholesterol,
            'Heart Rate': Heart_Rate,
            'Diabetes': Diabetes,
            'Family History': Family_History,
            'Smoking': Smoking,
            'Obesity': Obesity,
            'Alcohol Consumption': Alcohol_Consumption,
            'Exercise Hours Per Week': Exercise_Hours,
            'Diet': Diet,
            'Previous Heart Problems': Previous_Heart_Problems,
            'Medication Use': Medication_Use,
            'Stress Level': Stress_Level,
            'Sedentary Hours Per Day': Sedentary_Hours,
            'Income': Income,
            'BMI': BMI,
            'Triglycerides': Triglycerides,
            'Physical Activity Days Per Week': Physical_Activity_Days,
            'Sleep Hours Per Day': Sleep_Hours,
            'Systolic': Systolic,
            'Diastolic': Diastolic
        }
        # Make a prediction using the chosen model.
        prediction = predict_diagnosis(datapoint, model, scalers)
        if prediction == 1:
            st.success("Prediction: High risk of heart attack")
        else:
            st.success("Prediction: Low risk of heart attack")
    

def about_page():
    st.title("About This App")
    st.write("""
    This app uses a machine learning model to predict the risk of a heart attack.
    The data is preprocessed by splitting 'Blood Pressure' into 'Systolic' and 'Diastolic', mapping categorical values,
    and dropping unnecessary columns. Choose a model, enter patient details, and get a prediction.
    """)
    
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Predict", "About"])
    
    if page == "Predict":
        prediction_page()
    else:
        about_page()
        

if __name__ == '__main__':
    main()

import os
st.write("Current working directory:", os.getcwd())
st.write("Scalers path exists:", os.path.exists(scalers_path))
