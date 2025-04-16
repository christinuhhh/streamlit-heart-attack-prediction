import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. Load the preprocessed dataset
preprocessed_csv = "preprocessed.csv"
df = pd.read_csv(preprocessed_csv)

# 2. Define features and target variable
features = [
    'Age', 'Sex', 'Cholesterol', 'Heart Rate', 'Diabetes', 'Family History',
    'Smoking', 'Obesity', 'Alcohol Consumption', 'Exercise Hours Per Week', 'Diet',
    'Previous Heart Problems', 'Medication Use', 'Stress Level', 'Sedentary Hours Per Day',
    'Income', 'BMI', 'Triglycerides', 'Physical Activity Days Per Week', 'Sleep Hours Per Day',
    'Systolic', 'Diastolic'
]
target = 'Heart Attack Risk'

X = df[features]
y = df[target]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)

# Work on copies to avoid SettingWithCopy warnings
X_train = X_train.copy()
X_test = X_test.copy()

# 4. Scale features using MinMaxScaler (one scaler per column)
scalers = {}
for col in X_train.columns:
    scaler = MinMaxScaler()
    X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
    X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))
    scalers[col] = scaler

# 5. Train Logistic Regression model
logistic_model = LogisticRegression(class_weight='balanced', random_state=6)
logistic_model.fit(X_train, y_train)

# Evaluate Logistic Regression
y_pred_logistic = logistic_model.predict(X_test)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print("Logistic Regression Accuracy: {:.4f}".format(accuracy_logistic))
print("Classification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_logistic))
print("Confusion Matrix (Logistic Regression):")
print(confusion_matrix(y_test, y_pred_logistic))

# 6. Train Random Forest model
rf_model = RandomForestClassifier(random_state=6, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Evaluate Random Forest
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy: {:.4f}".format(accuracy_rf))
print("Classification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))

# 7. Create directories for saving models and scalers if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("scalers", exist_ok=True)

# 8. Save the models and scalers using joblib
joblib.dump(logistic_model, "models/logistic_regression_model.pkl")
joblib.dump(rf_model, "models/random_forest_model.pkl")
joblib.dump(scalers, "scalers/scalers.pkl")

print("Models and scalers saved successfully!")
