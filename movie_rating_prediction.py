# Task 2 - Movie Rating Prediction
# CodSoft Data Science Internship

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# load dataset
df = pd.read_csv('IMDb Movies India.csv', encoding='latin1')

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nSample rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# --- Cleaning ---

# drop rows where Rating is missing - that's our target
df.dropna(subset=['Rating'], inplace=True)

# Votes has commas like "1,234" - remove them and convert
df['Votes'] = df['Votes'].astype(str).str.replace(',', '').str.strip()
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
df['Votes'].fillna(0, inplace=True)

# Duration has "min" text - remove it
df['Duration'] = df['Duration'].astype(str).str.replace('min', '').str.strip()
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
df['Duration'].fillna(df['Duration'].median(), inplace=True)

# extract 4-digit year from Year column
df['Year'] = df['Year'].astype(str).str.extract('(\d{4})')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Year'].fillna(df['Year'].median(), inplace=True)

# --- Feature Engineering ---
# frequency encoding - how many movies does each director/genre/actor have
# more appearances = more well-known = useful signal

for col in ['Director', 'Genre', 'Actor 1']:
    df[col] = df[col].fillna('Unknown')
    freq_map = df[col].value_counts()
    df[col + '_freq'] = df[col].map(freq_map)

print("\nSample after processing:")
print(df[['Rating', 'Votes', 'Duration', 'Year', 'Director_freq', 'Genre_freq', 'Actor 1_freq']].head())

# --- Model ---

features = ['Votes', 'Duration', 'Year', 'Director_freq', 'Genre_freq', 'Actor 1_freq']
X = df[features]
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluation ---

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Results ---")
print(f"Mean Absolute Error : {mae:.3f}")
print(f"R² Score            : {r2:.3f}")

# feature importance
importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature Importance:")
print(importance)

# show some actual vs predicted
results = pd.DataFrame({
    'Actual Rating': y_test.values[:10],
    'Predicted Rating': y_pred[:10].round(2)
})
print("\nSample Predictions:")
print(results.to_string(index=False))
