# Credit Card Fraud Detection using XGBoost

## Overview

This project detects fraudulent credit card transactions using Machine Learning. The dataset is highly imbalanced, so SMOTE (Synthetic Minority Oversampling Technique) is used to balance the classes before training an XGBoost classifier.

The trained model is integrated with a Flask API for real-time fraud prediction.

## Features

* Data Preprocessing and Cleaning
* Exploratory Data Analysis (EDA)
* Fraud Detection using XGBoost
* Class Balancing using SMOTE
* Model Evaluation using Accuracy, Classification Report, Confusion Matrix, and ROC-AUC Score
* Model Persistence using Joblib
* Flask API for Real-Time Predictions
* GitHub Version Control

## Dataset

* Source: Kaggle Credit Card Fraud Detection Dataset
* Total Transactions: 284,807
* Fraudulent Transactions: 492
* Fraud Percentage: 0.17%

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Imbalanced-Learn (SMOTE)
* Flask
* Matplotlib
* Seaborn
* Joblib
* Git & GitHub

## Model Performance

| Metric          | Score  |
| --------------- | ------ |
| Accuracy        | 99.98% |
| ROC-AUC Score   | 99.98% |
| False Negatives | 0      |

## Author

Rohith Kuriti
