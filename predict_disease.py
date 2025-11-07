# predict_disease.py

import pickle
import numpy as np

# Load trained models
nb_model = pickle.load(open("naive_bayes.pkl", "rb"))
knn_model = pickle.load(open("knn.pkl", "rb"))
log_model = pickle.load(open("logistic.pkl", "rb"))

# Symptom list
symptoms = [
    "fever", "cough", "fatigue", "headache", "nausea",
    "weight_loss", "chest_pain", "yellow_skin", "body_pain", "sweating"
]

print("\n=== Disease Prediction System ===")
print("Enter 1 if symptom is present, else 0.\n")

user_input = []
for symptom in symptoms:
    val = int(input(f"Do you have {symptom}? (1/0): "))
    user_input.append(val)

user_input = np.array(user_input).reshape(1, -1)

# Make predictions
nb_pred = nb_model.predict(user_input)[0]
knn_pred = knn_model.predict(user_input)[0]
log_pred = log_model.predict(user_input)[0]

print("\n=== Prediction Results ===")
print(f"Naive Bayes Model Prediction:      {nb_pred}")
print(f"K-Nearest Neighbors Prediction:    {knn_pred}")
print(f"Logistic Regression Prediction:    {log_pred}")
