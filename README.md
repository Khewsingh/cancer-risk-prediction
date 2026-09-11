# Cancer Risk Prediction Using Machine Learning

A machine learning project that predicts an individual's cancer risk level as **Low, Medium, or High** using demographic, lifestyle, environmental, genetic, and medical factors.

## 📌 Project Overview

The objective of this project is to develop a multiclass machine learning model that can predict cancer risk levels from health-related features.

The project also focuses on handling **class imbalance** and improving the prediction of minority **High-risk** patients.

## 🎯 Problem Statement

The model predicts three cancer risk levels:

- Low Risk
- Medium Risk
- High Risk

The dataset contains demographic, lifestyle, environmental, genetic, and medical features.

## 📊 Features Used

### Demographic Features
- Age
- Gender
- BMI

### Lifestyle & Environmental Features
- Smoking
- Alcohol Use
- Obesity
- Red Meat Diet
- Salted/Processed Food
- Fruit & Vegetable Intake
- Physical Activity
- Physical Activity Level
- Air Pollution
- Occupational Hazards
- Calcium Intake

### Genetic & Medical Features
- Family History
- BRCA Mutation
- H. Pylori Infection

## 🔎 Exploratory Data Analysis

The analysis showed that the **Medium-risk class is the majority class**, while the High-risk class is the minority class.

Important factors associated with higher risk included:

- Air Pollution
- Smoking
- Alcohol Use
- Salted/Processed Food
- Occupational Hazards
- Red Meat Diet
- Obesity

BMI was not found to have a strong effect on the risk level in the analysis.

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

1. Removed `patient_id` because it was not useful for prediction.
2. Removed `cancer_type` because it was strongly correlated with the target.
3. Removed `Overall_Risk_Score` after identifying **data leakage**.
4. Label encoded the target variable.
5. Addressed class imbalance using **SMOTE** in selected experiments.

## 🤖 Machine Learning Models

Several approaches were experimented with:

- Logistic Regression
- Random Forest
- Random Forest + SMOTE
- Optuna-tuned Random Forest
- XGBoost
- XGBoost + SMOTE
- Class-weighted XGBoost
- Optuna-tuned Class-weighted XGBoost

## 🔧 Hyperparameter Optimization

**Optuna** was used for hyperparameter optimization to improve model performance.

The experiments focused not only on overall accuracy but also on improving recall for the minority High-risk class.

## 🏆 Final Model

The final selected model is:

**Optuna-Tuned Class-Weighted XGBoost**

Reported performance:

| Metric | Score |
|---|---:|
| Accuracy | 0.88 |
| Macro F1 | 0.72 |
| Weighted F1 | 0.87 |
| High-risk Recall | 0.45 |
| Low-risk Recall | 0.78 |
| Medium-risk Recall | 0.92 |

The final model was selected because it provided the best overall balance between accuracy and class-wise performance.

## 📁 Project Structure

```text
cancer-risk-prediction/
│
├── app.py
├── main.py
├── Cancer_Risk_Prediction_(ML).ipynb
├── cancer-risk-factors.csv
├── final_xgb_class_weighted.pkl
├── feature_names.pkl
├── label_encoder.pkl
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
