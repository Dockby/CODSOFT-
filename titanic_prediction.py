# Task 1 - Titanic Survival Prediction
# CodSoft Data Science Internship

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# load the dataset
df = pd.read_csv('tested.csv')

print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# --- Data Cleaning ---

# fill missing age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# fill embarked with the most common port
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# fill missing fare with median
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# cabin has way too many nulls, just drop it
df.drop(columns=['Cabin'], inplace=True)

# drop columns that won't help in prediction
df.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)

# convert sex to 0 and 1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# convert embarked to numbers
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

print("\nCleaned data sample:")
print(df.head())
print("\nAny nulls remaining?", df.isnull().sum().sum())

# --- Features and Target ---

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

# --- Training ---

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# --- Evaluation ---

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n--- Results ---")
print(f"Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# survival rate in the full dataset
print("Overall survival rate:", round(df['Survived'].mean() * 100, 2), "%")

# which features had the most impact
feature_coef = pd.Series(model.coef_[0], index=X.columns)
print("\nFeature Coefficients:")
print(feature_coef.sort_values(ascending=False))
