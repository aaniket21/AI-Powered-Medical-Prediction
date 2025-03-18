import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# ✅ Use the correct file path
file_path = r'D:\ai medical project 2\datasets\breast_cancer.csv'
data = pd.read_csv(file_path)

# Replace 'diagnosis' with the correct target column name
target_column = 'diagnosis'  # Change this based on your dataset

# Select the 5 features
selected_features = [
    'radius_mean', 'texture_mean', 'smoothness_mean', 
    'compactness_mean', 'concavity_mean'
]

# Ensure the target column exists
if target_column not in data.columns:
    raise KeyError(f"Target column '{target_column}' not found in the dataset!")

X = data[selected_features]
y = data[target_column]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(model, r'D:\ai medical project 2\models\breast_cancer_model.pkl')
joblib.dump(scaler, r'D:\ai medical project 2\models\breast_cancer_scaler.pkl')

print("Model and scaler saved successfully!")
