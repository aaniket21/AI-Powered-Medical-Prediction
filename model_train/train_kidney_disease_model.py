import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load dataset
file_path = "datasets/kidney_disease.csv"
data = pd.read_csv(file_path)

# Data Preprocessing
# Remove ID column
data = data.drop(['id'], axis=1)

# Convert object columns with numeric values to float
for col in ['pcv', 'wc', 'rc']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Encode target column
data['classification'] = data['classification'].map({'ckd': 1, 'notckd': 0})

# Fill missing values with column median for numeric and mode for categorical
for col in data.columns:
    if data[col].dtype == 'object':
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].median(), inplace=True)

# One-hot encode categorical columns
data = pd.get_dummies(data, drop_first=True)

# Splitting features and target
X = data.drop('classification', axis=1)
y = data['classification']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Save the model and scaler
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)
joblib.dump(model, os.path.join(output_dir, "kidney_disease_model.pkl"))
joblib.dump(scaler, os.path.join(output_dir, "kidney_disease_scaler.pkl"))

print(f"Model and scaler saved successfully in '{output_dir}'!")
