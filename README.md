# AHP + XGBoost Wellness Prediction

This branch contains a standalone implementation that predicts future readiness
scores using an AHP-weighted wellness score and n-1 supervised learning with XGBoost.

## Method Overview
- AHP assigns importance to wellness factors
- Factors are aggregated into a wellness score
- Past wellness history predicts future readiness

## Model
- XGBoost Regressor
- Look-back window: 5 days
- Evaluation metric: RMSE

This is an experimental implementation kept separate from main.