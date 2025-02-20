# Salary Prediction Model using Ordinal Logistic Regression

## Overview
This project focuses on predicting salary classes based on various features, such as age, gender, education level, years of experience, and others. The dataset comes from a Kaggle competition and contains responses to multiple survey questions, which have been cleaned and processed for use in machine learning tasks. The goal is to implement a model that can predict salary ranges (encoded in `Q29_Encoded`) using the features available.

## Contents
- **Data Cleaning**: Preparation of the dataset by removing irrelevant rows/columns and handling missing values.
- **Feature Engineering**: Encoding categorical features using appropriate techniques like one-hot and ordinal encoding.
- **Modeling**: Implementation of an Ordinal Logistic Regression model to handle ordinal classification tasks.
- **Hyperparameter Tuning**: Optimization of model performance using grid search to find the best hyperparameters.
- **Evaluation**: Performance evaluation using cross-validation, F1 scores, and visualization of feature importance.

## Requirements
- Python 3.x
- Libraries: 
  - pandas
  - numpy
  - seaborn
  - matplotlib
  - scikit-learn
  - Google Colab (for file uploads)

You can install the necessary libraries via pip:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

## Data
The data used in this project is a Kaggle survey dataset, which includes features related to the respondent's age, gender, education, country, salary range, and other professional information. The original dataset has 298 columns, which are reduced during the data cleaning process.

## Steps
### 1. Data Cleaning
The data cleaning process involves:
- Dropping irrelevant rows and columns (e.g., question details and columns with "Other" responses).
- Filling or dropping missing values for various columns.
- Imputing missing values with appropriate strategies such as "unknown" or mode imputation.

### 2. Feature Engineering
Different strategies are applied to encode the categorical features:
- Ordinal encoding for features with a natural ranking (e.g., age, education level).
- One-hot encoding for features without a natural ranking (e.g., gender, job title).
- Grouping countries into income groups based on World Bank classifications.

### 3. Model Implementation
An Ordinal Logistic Regression model is implemented from scratch to handle ordinal classification tasks:
- Multiple binary logistic regression models are used to predict class probabilities.
- The model computes the probability of each salary class and makes predictions based on these probabilities.

### 4. Model Evaluation
Cross-validation (10-fold) is performed to evaluate the model's performance:
- Accuracy and F1 scores are computed across folds.
- Bias-variance trade-off is analyzed by testing different values for regularization hyperparameter `C`.

### 5. Hyperparameter Tuning
Grid search is applied to find the best hyperparameters for the Ordinal Logistic Regression model, particularly focusing on the regularization parameter `C` and solver types.

### 6. Feature Importance
The model's coefficients are used to assess feature importance, and a bar plot is generated to show the most influential features for salary prediction.

## Results
The best-performing model achieves an average F1 score of **0.134** across 10-fold cross-validation, though model performance may be limited due to class imbalance in the target variable. For better results, techniques like class balancing (e.g., oversampling the minority class) could be considered.

### Visualizations:
- **Feature Importance Plot**: Shows which features are most influential in predicting salary.
- **Bias-Variance Trade-Off**: A graph showing how the regularization parameter affects the model's bias and variance.

## Conclusion
This project implements an Ordinal Logistic Regression model for salary prediction, utilizing feature engineering, hyperparameter tuning, and model evaluation. Further improvements can be made by addressing class imbalance and exploring additional feature engineering techniques.
