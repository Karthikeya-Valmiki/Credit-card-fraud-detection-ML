# Credit Card Fraud Detection Using Machine Learning

## Project Overview

Credit card fraud is a significant issue that impacts both financial institutions and consumers. Detecting fraudulent transactions efficiently and accurately is essential to mitigate losses and ensure financial security. This project, titled *"Credit Card Fraud Detection Using Machine Learning"*, explores the use of various machine learning techniques to identify fraudulent transactions from a credit card dataset.

In this project, we employ an end-to-end approach to build a fraud detection system, starting with data preprocessing, feature engineering, model training, and evaluation. The dataset, *"transactions.txt"*, contains transaction details along with labels indicating whether the transaction is fraudulent or not. We utilize several machine learning models, including Logistic Regression, Random Forest, Gradient Boosting, and Neural Networks, to achieve the best performance in detecting fraudulent transactions.

## Repository Contents

This repository includes the following files and directories:

- **`credit_card_fraud_detection.ipynb`**: The main Jupyter Notebook that contains the full implementation of the project. It covers:
  - Importing libraries and loading the dataset
  - Exploratory Data Analysis (EDA)
  - Data preprocessing (missing value handling, feature engineering, one-hot encoding)
  - Handling imbalanced data using SMOTE and undersampling
  - Model training and evaluation with various classifiers
  - Hyperparameter tuning using GridSearchCV and RandomizedSearchCV
  - Performance evaluation and final model selection
  
- **`data link.txt`**: The .txt file which contains a kaggle link, where you can download the data file - "transactions.txt". Dataset used in the project contains transaction details and labels for fraud detection. The dataset includes both numeric and categorical features.

- **`requirements.txt`**: A file listing all the required Python libraries and dependencies for running the project, including machine learning and data manipulation packages such as `scikit-learn`, `keras`, `imblearn`, and `pandas`.

- **`README.md`**: This file, providing an overview of the project, repository contents, and instructions for use.

## Steps Performed in the Project

1. **Loading Libraries and Data**: Importing the necessary libraries and loading the dataset.
2. **Exploratory Data Analysis (EDA)**: 
   - Descriptive statistics and data visualizations to understand the dataset.
   - Identifying missing data and performing necessary data cleaning steps.
3. **Variable Study**: Examining and handling individual features in the dataset.
4. **Feature Engineering**: One-hot encoding of categorical variables and creating new features if necessary.
5. **Handling Imbalanced Data**: Using techniques like SMOTE (Synthetic Minority Over-sampling Technique) and random undersampling to address class imbalance.
6. **Model Selection**: Training and evaluating multiple machine learning models, including:
   - Logistic Regression
   - Decision Trees
   - Random Forest Classifier
   - Gradient Boosting Classifier
   - Support Vector Classifier (SVC)
   - Neural Networks
7. **Model Evaluation**: Hyperparameter tuning using GridSearchCV and RandomizedSearchCV, and evaluating model performance using classification metrics like accuracy, precision, recall, and F1 score.

## Author

**Karthikeya Valmiki**

## How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Karthikeya-Valmiki/Credit-card-fraud-detection-ML.git
