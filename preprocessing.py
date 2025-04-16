import pandas as pd

# 1. Load the raw dataset
csv_file = "heart_attack_prediction_dataset.csv"  # Adjust this path if needed
df = pd.read_csv(csv_file)

# 2. Data Cleaning & Preprocessing

# Convert 'Blood Pressure' into two separate numerical columns 'Systolic' and 'Diastolic'
df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)

# Convert 'Sex' categorical column into numerical representation ('Male' -> 1, 'Female' -> 0)
df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})

# Convert 'Diet' categorical column into numerical representation ('Unhealthy' -> 0, 'Average' -> 1, 'Healthy' -> 2)
df['Diet'] = df['Diet'].map({'Unhealthy': 0, 'Average': 1, 'Healthy': 2})

# Drop unnecessary columns that do not contribute to analysis
df.drop(columns=['Patient ID', 'Country', 'Continent', 'Hemisphere', 'Blood Pressure'], inplace=True)

# Display info to verify preprocessing
print("DataFrame Info after Preprocessing:")
print(df.info())

# Save the preprocessed CSV so that it can be used for training and in your app
preprocessed_csv = "preprocessed.csv"
df.to_csv(preprocessed_csv, index=False)
print(f"Preprocessed data saved to {preprocessed_csv}")
