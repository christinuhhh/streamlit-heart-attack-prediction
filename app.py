import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Define a global scalers path for debugging purposes
GLOBAL_SCALERS_PATH = os.path.join("scalers", "scalers.pkl")

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
    # Use the local scalers path variable (could also define this inside the function)
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
    
    # Create a form for input features with two columns
    with st.form("prediction_form"):
        # Create two columns
        col1, col2 = st.columns(2)

        # --------- Column 1: Physical & Medical Attributes ---------
        with col1:
            Age = st.number_input("Age", min_value=0, max_value=120, value=50, step=1)
            Sex_input = st.selectbox("Sex", options=["Male", "Female"])
            Sex = 1 if Sex_input == "Male" else 0
            Cholesterol = st.number_input("Cholesterol", min_value=0.0, value=200.0)
            Heart_Rate = st.number_input("Heart Rate", min_value=0, value=70, step=1)
            Diabetes_input = st.selectbox("Diabetes", options=["No", "Yes"])
            Diabetes = 1 if Diabetes_input == "Yes" else 0
            Family_History_input = st.selectbox("Family History", options=["No", "Yes"])
            Family_History = 1 if Family_History_input == "Yes" else 0
            BMI = st.number_input("BMI", min_value=0.0, value=24.0, step=0.1)
            Systolic = st.number_input("Systolic", min_value=0, max_value=300, value=120, step=1)
            Diastolic = st.number_input("Diastolic", min_value=0, max_value=200, value=80, step=1)
            Previous_Heart_Problems_input = st.selectbox("Previous Heart Problems", options=["No", "Yes"])
            Previous_Heart_Problems = 1 if Previous_Heart_Problems_input == "Yes" else 0
            Medication_Use_input = st.selectbox("Medication Use", options=["No", "Yes"])
            Medication_Use = 1 if Medication_Use_input == "Yes" else 0

        # --------- Column 2: Lifestyle Attributes ---------
        with col2:
            Smoking_input = st.selectbox("Smoking", options=["No", "Yes"])
            Smoking = 1 if Smoking_input == "Yes" else 0
            Obesity_input = st.selectbox("Obesity", options=["No", "Yes"])
            Obesity = 1 if Obesity_input == "Yes" else 0
            Alcohol_Consumption = st.number_input("Alcohol Consumption (units per week)", min_value=0.0, value=0.0, step=0.5)
            Exercise_Hours = st.number_input("Exercise Hours Per Week", min_value=0.0, value=3.0, step=0.5)
            Diet_input = st.selectbox("Diet", options=["Unhealthy", "Average", "Healthy"])
            if Diet_input == "Unhealthy":
                Diet = 0
            elif Diet_input == "Average":
                Diet = 1
            else:
                Diet = 2
            Stress_Level = st.number_input("Stress Level (0-10)", min_value=0, max_value=10, value=5, step=1)
            Sedentary_Hours = st.number_input("Sedentary Hours Per Day", min_value=0.0, value=8.0, step=0.5)
            Income = st.number_input("Annual Income", min_value=0.0, value=50000.0, step=1000.0)
            Triglycerides = st.number_input("Triglycerides", min_value=0.0, value=150.0, step=1.0)
            Physical_Activity_Days = st.number_input("Physical Activity Days Per Week", min_value=0, max_value=7, value=3, step=1)
            Sleep_Hours = st.number_input("Sleep Hours Per Day", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        
        submitted = st.form_submit_button("Predict")
    
    if submitted:
        # Assemble all input values into a dictionary matching the feature list order.
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
        prediction = predict_diagnosis(datapoint, model, scalers)
        if prediction == 1:
            st.error("Prediction: High risk of heart attack")
        else:
            st.success("Prediction: Low risk of heart attack")



def about_page():
    st.title("About This App")
    st.write("""
    This app uses machine learning models to predict the risk of a heart attack. The data is preprocessed by splitting 
    'Blood Pressure' into 'Systolic' and 'Diastolic', converting text values to numbers, and removing unnecessary information.
    You can choose a model, enter your health details, and then get a prediction.
    """)

    st.write("Below are simple explanations of the two models used:")

    # Create two columns for the explanations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Logistic Regression")
        st.markdown("""
        **What it does:**  
        Think of Logistic Regression like a simple calculator that weighs different factors—such as age, blood pressure, and cholesterol—to give you a risk number between 0 (low risk) and 1 (high risk).  
        
        **Why use it:**  
        It provides one clear answer based on a weighted combination of your health details. It’s simple and easy to understand.
        """)
    
    with col2:
        st.subheader("Random Forest")
        st.markdown("""
        **What it does:**  
        Random Forest is like asking a group of doctors for their opinions. It builds many small decision trees and then takes a vote on what the overall risk should be.  
        
        **Why use it:**  
        By considering the opinions of many models, it can handle more complex information and is often more robust in its predictions.
        """)

    st.write("Both models provide useful insights into your heart attack risk, but they take slightly different approaches to the prediction.")


def main():
    st.sidebar.title("Heart Attack Prediction App")
    page = st.sidebar.radio("Go to", ["Predict", "About"])
    
    if page == "Predict":
        prediction_page()
    else:
        about_page()

if __name__ == '__main__':
    main()
