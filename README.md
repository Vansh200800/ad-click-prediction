# Ad Click Prediction — ML Classification with ANN

**Course:** Data Science Portfolio — Macquarie University  
**Tools:** Python · Pandas · Scikit-learn · TensorFlow · Seaborn · Jupyter Notebook

---

## Project Overview
Built a complete machine learning pipeline to predict whether an internet user will click on a digital advertisement — a core problem in digital marketing analytics. Progressed from EDA through clustering, logistic regression with hyperparameter tuning, to an Artificial Neural Network achieving **98% accuracy**.

---

## Business Problem
A digital marketing team needs to identify which users are most likely to click on ads, enabling targeted advertising and reducing wasted ad spend. This model predicts click behaviour based on user demographics and online activity.

---

## Dataset
- **Source:** Kaggle — Ad Click Dataset
- **Size:** 1,000 users, 10 features
- **Target:** `Clicked on Ad` (1 = clicked, 0 = did not click)
- **Features:** Daily Time on Site, Age, Area Income, Daily Internet Usage, City, Country, Gender

---

## Methodology

### 1. Data Cleaning & EDA
- Handled missing values: mean imputation for numeric, forward-fill for categorical
- Visualised distributions of age, income, internet usage, and time on site
- Correlation heatmap to identify feature relationships

### 2. Customer Segmentation (K-Means Clustering)
- Applied elbow method → identified **4 optimal customer clusters** based on Age and Area Income
- Visualised distinct customer segments for marketing strategy

### 3. Feature Selection
- Used ExtraTreesClassifier to rank feature importances
- Selected top 4 features: Daily Time on Site, Age, Area Income, Daily Internet Usage

### 4. Logistic Regression + GridSearchCV
- K-Fold Cross Validation (5-fold)
- GridSearchCV over C values and solvers → **Best accuracy: ~97%**
- Optimal parameters: C=0.5, solver='newton-cg'

### 5. Artificial Neural Network (ANN)
- Architecture: Input → Dense(6, ReLU) → Dense(6, ReLU) → Dense(1, Sigmoid)
- Optimizer: Adam | Loss: Binary Crossentropy | Epochs: 100
- **Final ANN Accuracy: 98%**

---

## Results Summary

| Model | Accuracy |
|---|---|
| Logistic Regression (baseline) | ~95% |
| Logistic Regression + GridSearchCV | ~97% |
| Artificial Neural Network | ~98% |

---

## Key Findings
- Daily Internet Usage and Daily Time on Site are the strongest predictors of ad clicks
- Younger users with lower internet usage and lower area income are less likely to click
- ANN marginally outperforms optimised Logistic Regression — but LR is preferred for interpretability in production

---

## Tech Stack


## Author
**Vansh Sondhi** | [github.com/Vansh200800](https://github.com/Vansh200800)
