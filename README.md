# Task 4 – Sales Prediction using Python

## Objective
Predict the sales of a product based on advertising budgets spent on TV, Radio, and Newspaper channels.

## Dataset
- Source: [Kaggle – Advertising Dataset](https://www.kaggle.com/datasets/ashydv/advertising-dataset)
- File needed: `advertising.csv`
- Place it in this folder before running.

## Approach
- Explored correlation between each advertising channel and sales
- Trained a Linear Regression model using TV, Radio, and Newspaper spend as features
- Evaluated using MAE, RMSE, and R² score
- Added a manual prediction example at the end

## Results
- **Model:** Linear Regression
- **R² Score:** ~0.90
- **Key finding:** TV and Radio spending are the strongest predictors of sales. Newspaper has very little impact.

## How to Run
```bash
python sales_prediction.py
```
