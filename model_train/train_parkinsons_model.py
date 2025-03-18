import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Load the dataset
file_path = "datasets/parkinsons.csv"
data = pd.read_csv(file_path)

# Data Preprocessing
X = data.drop(["name", "status"], axis=1)  # Features
y = data["status"]                         # Target

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the RandomForest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\n✅ Model Accuracy:", accuracy)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🔍 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model and scaler
output_folder = "models"
os.makedirs(output_folder, exist_ok=True)

model_path = os.path.join(output_folder, "parkinsons_model.pkl")
scaler_path = os.path.join(output_folder, "parkinsons_scaler.pkl")

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"\n🚀 Model saved at: {model_path}")
print(f"🔧 Scaler saved at: {scaler_path}")
