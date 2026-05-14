# Task 2 – Movie Rating Prediction

## Objective
Build a regression model to predict the rating of an Indian movie based on features like genre, director, and actors.

## Dataset
- Source: [Kaggle – IMDb India Movies](https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies)
- File needed: `IMDb Movies India.csv`
- Place it in this folder before running.

## Approach
- Cleaned Votes (removed commas), Duration (removed "min"), and extracted Year from string
- Used **frequency encoding** for Director, Genre, and Actor 1 — maps each to how many movies they appear in
- Trained a Random Forest Regressor
- Evaluated using MAE and R² score

## Results
- **Model:** Random Forest Regressor
- **MAE:** ~0.5 (predictions are within ~0.5 rating points on average)
- **Key finding:** Number of votes and director frequency had the highest influence on predicted rating


