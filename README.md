# Loan Eligibility Prediction

A machine learning classifier that predicts whether a home loan application will be approved, based on applicant income, credit history, education, property area and other profile details.

## Problem

Lenders review large volumes of applications manually. Automating a first-pass eligibility check lets them prioritize likely approvals and target the right customer segments.

## Approach

1. **Exploratory analysis:** income distributions and outliers, income by gender and education, approval rates by property area
2. **Data preparation:** mode and mean imputation for missing values, a combined total income feature, log transforms on loan amount and income to reduce skew, label encoding and feature scaling
3. **Modelling:** trained and compared a Decision Tree and a Gaussian Naive Bayes classifier
4. **Scoring:** applied the best model to an unseen test set

## Results

| Model | Accuracy |
|---|---|
| Decision Tree | 70.7% |
| **Gaussian Naive Bayes** | **82.9%** |

Naive Bayes performed best on the held-out data.

## Tech stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn

## Files

| File | Description |
|---|---|
| `loan_eligibility_prediction.ipynb` | Full analysis and models |
| `loan-train.csv` / `loan-test.csv` | Training and test data |
