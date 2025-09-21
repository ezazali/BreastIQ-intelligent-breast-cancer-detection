import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load CSV
df = pd.read_csv('/kaggle/input/anesskhan/Breast_Cancer.csv')  # Replace with your CSV file path
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces

# Drop missing values
df.dropna(inplace=True)

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and targets
X = df.drop(columns=['T Stage', 'Status'])
y_t = df['T Stage']      # Target 1: T Stage
y_s = df['Status']       # Target 2: Status

# 80:20 split
X_train, X_test, y_train_s, y_test_s = train_test_split(X, y_s, test_size=0.2, random_state=42)
_, _, y_train_t, y_test_t = train_test_split(X, y_t, test_size=0.2, random_state=42)

# Models for Status prediction
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

print("\n--- Prediction Results for 'Status' (Alive/Dead) ---")
for name, model in models.items():
    model.fit(X_train, y_train_s)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test_s, y_pred)
    prec = precision_score(y_test_s, y_pred, average='binary')
    rec = recall_score(y_test_s, y_pred, average='binary')
    f1 = f1_score(y_test_s, y_pred, average='binary')

    print(f"\nModel: {name}")
    print(f"Accuracy:  {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall:    {rec:.2f}")
    print(f"F1 Score:  {f1:.2f}")
    print("Classification Report:")
    print(classification_report(y_test_s, y_pred))

# -----------------------------------------------------
# Predicting T Stage as Classification (e.g., T1/T2/T3)
# -----------------------------------------------------
print("\n--- Prediction Results for 'T Stage' ---")
clf_rf_t = RandomForestClassifier(random_state=42)
clf_rf_t.fit(X_train, y_train_t)
y_pred_t = clf_rf_t.predict(X_test)

print("Random Forest Classifier (T Stage):")
print("Accuracy:", accuracy_score(y_test_t, y_pred_t))
print("Classification Report:")
print(classification_report(y_test_t, y_pred_t))
