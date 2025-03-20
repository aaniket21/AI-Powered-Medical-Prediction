import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
file_path = "datasets/heart_disease_uci.csv"  # Update path if needed
data = pd.read_csv(file_path)

# Auto-detect target column
target_column = "num"  # The actual target column in your dataset
if target_column not in data.columns:
    raise KeyError(f"❌ Target column '{target_column}' not found!")

# Drop rows where the target column is missing
data = data.dropna(subset=[target_column])

# Encode categorical columns
categorical_columns = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
for col in categorical_columns:
    if col in data.columns:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))

# Handle missing values (fill with median)
data.fillna(data.median(numeric_only=True), inplace=True)

# Define features (excluding 'id' and target column)
features = [col for col in data.columns if col not in ['id', target_column]]
X = data[features]
y = data[target_column].apply(lambda x: 1 if x > 0 else 0)  # Convert multi-class to binary

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
with open("models/heart_disease_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model training complete! Saved as 'heart_disease_model.pkl'")
