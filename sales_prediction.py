# Task 4 - Sales Prediction using Python
# CodSoft Data Science Internship

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# load dataset
# columns: TV, Radio, Newspaper, Sales
df = pd.read_csv('advertising.csv')

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nBasic Statistics:")
print(df.describe())

print("\nMissing values:", df.isnull().sum().sum())

# --- Correlation Analysis ---

print("\nCorrelation with Sales:")
print(df.corr()['Sales'].sort_values(ascending=False))

# TV has the strongest correlation, newspaper the weakest

# --- Model ---

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# --- Evaluation ---

y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\n--- Model Performance ---")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")

# coefficients tell us impact of each channel
print("\nCoefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.4f}")
print(f"  Intercept: {model.intercept_:.4f}")

print("\nInterpretation:")
print(f"  Every 1 unit increase in TV budget → Sales increase by {model.coef_[0]:.4f}")
print(f"  Every 1 unit increase in Radio budget → Sales increase by {model.coef_[1]:.4f}")
print(f"  Newspaper has the least effect on sales.")

# sample predictions
results = pd.DataFrame({
    'Actual Sales'   : y_test.values[:10],
    'Predicted Sales': y_pred[:10].round(2),
    'Difference'     : (y_test.values[:10] - y_pred[:10]).round(2)
})
print("\nSample Predictions:")
print(results.to_string(index=False))

# manual test
print("\n--- Manual Prediction ---")
sample = np.array([[230, 37, 69]])
pred = model.predict(sample)
print(f"TV=230, Radio=37, Newspaper=69 → Predicted Sales: {pred[0]:.2f}")
