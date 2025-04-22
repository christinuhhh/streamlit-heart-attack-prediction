import joblib

model = joblib.load("models/random_forest_model.pkl")
print(type(model))
