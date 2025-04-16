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
            scaled_value = scalers[col].transform(np.array([[value]])).flatten()[0]
            datapoint_list.append(scaled_value)
        else:
            st.error(f"Missing value for column: {col}")
            st.stop()
    datapoint_array = np.array(datapoint_list).reshape(1, -1)
    prediction = model.predict(datapoint_array)[0]
    return prediction


# ---------------------------
# Prediction Page
# ---------------------------
def prediction_page():
    st.title("Heart Attack Risk Prediction")
    st.write("Enter the patient details below and choose a model for prediction.")

    # Let user choose which model to use
    model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest"])
    
    # Load the chosen model and scalers (cached)
    model, scalers = load_models_and_scalers(model_choice)
    
    # Create a form for input features with two columns for a nicer layout.
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        # Column 1: Physical & Medical Attributes
        with col1:
            Age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state.get("Age", 50), step=1, key="Age")
            Sex = st.selectbox("Sex", options=["Male", "Female"], key="Sex")
            Cholesterol = st.number_input("Cholesterol", min_value=0.0, value=st.session_state.get("Cholesterol", 200.0), key="Cholesterol")
            Heart_Rate = st.number_input("Heart Rate", min_value=0, value=st.session_state.get("Heart_Rate", 70), step=1, key="Heart_Rate")
            Diabetes = st.selectbox("Diabetes", options=["No", "Yes"], key="Diabetes")
            Family_History = st.selectbox("Family History", options=["No", "Yes"], key="Family_History")
            BMI = st.number_input("BMI", min_value=0.0, value=st.session_state.get("BMI", 24.0), step=0.1, key="BMI")
            Systolic = st.number_input("Systolic", min_value=0, max_value=300, value=st.session_state.get("Systolic", 120), step=1, key="Systolic")
            Diastolic = st.number_input("Diastolic", min_value=0, max_value=200, value=st.session_state.get("Diastolic", 80), step=1, key="Diastolic")
            Previous_Heart_Problems = st.selectbox("Previous Heart Problems", options=["No", "Yes"], key="Previous_Heart_Problems")
            Medication_Use = st.selectbox("Medication Use", options=["No", "Yes"], key="Medication_Use")
        
        # Column 2: Lifestyle Attributes
        with col2:
            Smoking = st.selectbox("Smoking", options=["No", "Yes"], key="Smoking")
            Obesity = st.selectbox("Obesity", options=["No", "Yes"], key="Obesity")
            Alcohol_Consumption = st.number_input("Alcohol Consumption (units per week)", min_value=0.0, value=st.session_state.get("Alcohol_Consumption", 0.0), step=0.5, key="Alcohol_Consumption")
            Exercise_Hours = st.number_input("Exercise Hours Per Week", min_value=0.0, value=st.session_state.get("Exercise_Hours", 3.0), step=0.5, key="Exercise_Hours")
            Diet = st.selectbox("Diet", options=["Unhealthy", "Average", "Healthy"], key="Diet")
            Stress_Level = st.number_input("Stress Level (0-10)", min_value=0, max_value=10, value=st.session_state.get("Stress_Level", 5), step=1, key="Stress_Level")
            Sedentary_Hours = st.number_input("Sedentary Hours Per Day", min_value=0.0, value=st.session_state.get("Sedentary_Hours", 8.0), step=0.5, key="Sedentary_Hours")
            Income = st.number_input("Income", min_value=0.0, value=st.session_state.get("Income", 50000.0), step=1000.0, key="Income")
            Triglycerides = st.number_input("Triglycerides", min_value=0.0, value=st.session_state.get("Triglycerides", 150.0), step=1.0, key="Triglycerides")
            Physical_Activity_Days = st.number_input("Physical Activity Days Per Week", min_value=0, max_value=7, value=st.session_state.get("Physical_Activity_Days", 3), step=1, key="Physical_Activity_Days")
            Sleep_Hours = st.number_input("Sleep Hours Per Day", min_value=0.0, max_value=24.0, value=st.session_state.get("Sleep_Hours", 7.0), step=0.5, key="Sleep_Hours")
        
        submitted = st.form_submit_button("Predict")
    if submitted:
        # Convert selectbox values to numeric as needed.
        Sex_val = 1 if st.session_state["Sex"] == "Male" else 0
        Diabetes_val = 1 if st.session_state["Diabetes"] == "Yes" else 0
        Family_History_val = 1 if st.session_state["Family_History"] == "Yes" else 0
        Previous_Heart_Problems_val = 1 if st.session_state["Previous_Heart_Problems"] == "Yes" else 0
        Medication_Use_val = 1 if st.session_state["Medication_Use"] == "Yes" else 0
        Smoking_val = 1 if st.session_state["Smoking"] == "Yes" else 0
        Obesity_val = 1 if st.session_state["Obesity"] == "Yes" else 0
        Diet_val = 0 if st.session_state["Diet"] == "Unhealthy" else (1 if st.session_state["Diet"] == "Average" else 2)

        datapoint = {
            'Age': st.session_state["Age"],
            'Sex': Sex_val,
            'Cholesterol': st.session_state["Cholesterol"],
            'Heart Rate': st.session_state["Heart_Rate"],
            'Diabetes': Diabetes_val,
            'Family History': Family_History_val,
            'Smoking': Smoking_val,
            'Obesity': Obesity_val,
            'Alcohol Consumption': st.session_state["Alcohol_Consumption"],
            'Exercise Hours Per Week': st.session_state["Exercise_Hours"],
            'Diet': Diet_val,
            'Previous Heart Problems': Previous_Heart_Problems_val,
            'Medication Use': Medication_Use_val,
            'Stress Level': st.session_state["Stress_Level"],
            'Sedentary Hours Per Day': st.session_state["Sedentary_Hours"],
            'Income': st.session_state["Income"],
            'BMI': st.session_state["BMI"],
            'Triglycerides': st.session_state["Triglycerides"],
            'Physical Activity Days Per Week': st.session_state["Physical_Activity_Days"],
            'Sleep Hours Per Day': st.session_state["Sleep_Hours"],
            'Systolic': st.session_state["Systolic"],
            'Diastolic': st.session_state["Diastolic"]
        }
        prediction = predict_diagnosis(datapoint, model, scalers)
        if prediction == 1:
            st.success("Prediction: High risk of heart attack")
        else:
            st.success("Prediction: Low risk of heart attack")

# ---------------------------
# About Page
# ---------------------------
def about_page():
    st.title("About This App")
    st.write("""
    This app uses machine learning models to predict the risk of a heart attack.
    The data is preprocessed by splitting 'Blood Pressure' into 'Systolic' and 'Diastolic', converting text values to numbers,
    and removing unnecessary information. Choose a model, enter your health details, and get a prediction.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Logistic Regression")
        st.markdown("""
        **What it does:**  
        Acts like a simple calculator that weighs your health details to provide a risk number between 0 and 1.
        
        **Why use it:**  
        It’s straightforward and easy to understand.
        """)
    with col2:
        st.subheader("Random Forest")
        st.markdown("""
        **What it does:**  
        Builds many small decision trees (like asking a group of experts) and takes a vote on your risk.
        
        **Why use it:**  
        It can handle complex patterns and is often very reliable.
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

