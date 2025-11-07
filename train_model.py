# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("dataset.csv")

# Split features and labels
X = data.drop("disease", axis=1)
y = data["disease"]

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1️⃣ Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)
pickle.dump(nb_model, open("naive_bayes.pkl", "wb"))

# 2️⃣ K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)
pickle.dump(knn_model, open("knn.pkl", "wb"))

# 3️⃣ Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
log_acc = accuracy_score(y_test, log_pred)
pickle.dump(log_model, open("logistic.pkl", "wb"))

# Print accuracy results
print("\n=== Model Training Completed ===")
print(f"Naive Bayes Accuracy:       {nb_acc * 100:.2f}%")
print(f"K-Nearest Neighbors Accuracy: {knn_acc * 100:.2f}%")
print(f"Logistic Regression Accuracy: {log_acc * 100:.2f}%")

print("\n✅ Models have been trained and saved successfully!")
