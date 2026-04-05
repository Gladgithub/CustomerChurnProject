# Customer Churn Prediction

This project uses machine learning to predict whether a telecom customer is likely to churn. It applies data preprocessing, exploratory data analysis, and classification models to identify customer retention patterns and key churn drivers.

## Problem Statement

Customer churn is a major business challenge because losing existing customers can reduce revenue and increase acquisition costs. This project aims to:
- Predict whether a customer is likely to churn
- Identify the main factors associated with churn
- Generate insights that support customer retention strategies

## Tools and Technologies

- Python
- Pandas
- Matplotlib
- Scikit-learn

## Machine Learning Concepts Used

- Supervised Learning
- Binary Classification
- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Model Evaluation
- Feature Importance Analysis

## Workflow

1. Load telecom customer data  
2. Clean and preprocess the dataset  
3. Perform exploratory data analysis  
4. Encode categorical variables  
5. Split data into training and testing sets  
6. Train Logistic Regression and Random Forest models  
7. Evaluate model performance  
8. Analyze important features influencing churn  

## Key Insights

- Customers on month-to-month contracts churn at much higher rates than customers on longer-term contracts  
- Customers with shorter tenure are more likely to leave  
- Higher monthly charges are associated with increased churn risk  
- Approximately 26.5% of customers churn, indicating a significant retention challenge  

## Dataset

This project uses the publicly available Telco Customer Churn dataset from Kaggle:

https://www.kaggle.com/blastchar/telco-customer-churn

Download the dataset and place the CSV file in the project root directory with the name:

WA_Fn-UseC_-Telco-Customer-Churn.csv

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt