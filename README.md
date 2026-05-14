# Task 1 – Titanic Survival Prediction

## Objective
Predict whether a passenger survived the Titanic disaster based on features like age, gender, ticket class, and fare.

## Dataset
- Source: [Kaggle – Titanic Dataset](https://www.kaggle.com/datasets/brendan45774/test-file)
- File needed: `tested.csv`
- Place it in this folder before running the script.

## Approach
- Handled missing values in Age, Fare, and Embarked columns
- Dropped the Cabin column due to too many missing entries
- Encoded categorical columns (Sex, Embarked) as numbers
- Trained a Logistic Regression model on 80% of the data
- Evaluated on the remaining 20%

## Results
- **Model:** Logistic Regression
- **Accuracy:** ~80%
- **Key finding:** Gender (female) and passenger class (1st class) had the strongest positive influence on survival

