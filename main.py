# ===============================
# HEART DISEASE PREDICTION
# ===============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==============================
# STEP 1: LOAD DATA
# ==============================
df = pd.read_csv("Project\heart.csv")

# ==============================
# STEP 2: FUZZY TRANSFORM
# ==============================
def fuzzy_age(age):
    if age <= 40:
        return 0.3
    elif age <= 60:
        return 0.6
    else:
        return 0.9

def fuzzy_chol(chol):
    if chol <= 180:
        return 0.3
    elif chol <= 240:
        return 0.6
    else:
        return 0.9

def fuzzy_bp(bp):
    if bp <= 120:
        return 0.3
    elif bp <= 140:
        return 0.6
    else:
        return 0.9

# Apply fuzzy logic (modify column names if needed)
df["age"] = df["age"].apply(fuzzy_age)
df["chol"] = df["chol"].apply(fuzzy_chol)
df["trestbps"] = df["trestbps"].apply(fuzzy_bp)

# ==============================
# STEP 3: SPLIT DATA
# ==============================
X = df.drop("target", axis=1)
y = df["target"]

# ==============================
# STEP 4: TRAIN-TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ==============================
# STEP 5: NORMALIZATION
# ==============================
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# STEP 6: MODEL (NEURO PART)
# ==============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ==============================
# STEP 7: PREDICTION
# ==============================
y_pred = model.predict(X_test)

# ==============================
# STEP 8: METRICS
# ==============================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print(f"Accuracy  : {accuracy:.2f}")
print(f"Precision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")
print(f"F1 Score  : {f1:.2f}")

print("\n===== CONFUSION MATRIX =====")
print(cm)