# Stroke Risk Prediction using Machine Learning

This project aims to develop a machine learning model capable of predicting the risk of stroke in individuals based on their demographic and health-related attributes. The model is intended to assist healthcare professionals in early detection and timely intervention.

---

## Problem Statement

Stroke is a leading cause of death and long-term disability worldwide. Early detection and risk prediction can greatly improve patient outcomes. This project seeks to build a predictive model that classifies patients into high-risk or low-risk categories for stroke.

---

## Dataset

- **Source**: [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Attributes**:
  - `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`, `stroke`
- **Target Variable**: `stroke` (1 = stroke, 0 = no stroke)

---

## Project Structure

- `DataProcessing.ipynb`: Data cleaning, preprocessing, and feature selection.
- `Model.ipynb`: Model training, evaluation, and performance metrics.

---

## Data Processing Workflow

1. **Load dataset** from CSV file.
2. **Handle missing values**, especially in the `bmi` and `smoking_status` columns.
3. **Encode categorical features** using `LabelEncoder`.
4. **Rescale features** using `MinMaxScaler` or `StandardScaler`.
5. **Handle class imbalance** using `SMOTE` and `RandomUnderSampler`.
6. **Split dataset** into training and test sets (stratified).

---

## Modeling

Multiple classification models were evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Stacking Classifier

Model evaluation was performed using:

- Confusion Matrix
- Classification Report
- ROC-AUC Score
- Cross-Validation (Stratified K-Folds)

---

## Performance Metrics

Evaluation was based on:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **ROC-AUC**

The final ensemble (stacked) model showed improved generalization and robust performance.

---

## Requirements

Install dependencies using:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

