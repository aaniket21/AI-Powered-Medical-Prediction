import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
# 1. Load the Dataset
file_path = "datasets/heart_disease_uci.csv"
data = pd.read_csv(file_path)
data.fillna(data.median(numeric_only=True), inplace=True)

# Encode categorical variables
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Binarize the target column: 0 (no disease), 1+ (disease → 1)
data['target'] = (data['num'] > 0).astype(int)

# Separating features and labels
X = data.drop(['id', 'dataset', 'num'], axis=1)  # Remove unnecessary columns
y = data['target']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining Heart Disease Model...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
# 4. Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------
# 5. Save the Model
# -----------------------
output_path = "models/heart_disease_model.pkl"
joblib.dump(model, output_path)

print(f"\n✅ Model saved as {output_path}")
