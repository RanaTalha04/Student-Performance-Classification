# Student Performance Prediction

This project predicts whether a student will **pass or fail** based on personal, academic, and social attributes. The dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Student+Performance).

---

## Table of Contents

- [Dataset](#dataset)
- [Features](#features)
- [Preprocessing](#preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Saved Model](#saved-model)
- [Author](#author)

---

## Dataset

The dataset contains student information including:  

- Demographics: `age`, `sex`, `address`, etc.  
- Academic: `studytime`, `failures`, `G1`, `G2`, `G3` (grades)  
- Social & personal: `famsup`, `activities`, `internet`, `romantic`, etc.  

The target variable `pass_fail` is **binary**:  
- `1` → Pass (G3 >= 10)  
- `0` → Fail (G3 < 10)  

---

## Features

After preprocessing and feature selection, the following key features are used:

- `age`, `Medu`, `Fedu`, `studytime`, `failures`, `higher`, `Dalc`, `Walc`  
- Engineered features:  
  - `total_alcohol` = Dalc + Walc  
  - `study_fail_ratio` = studytime / (failures + 1)

---

## Preprocessing

- Convert yes/no columns to binary (1/0).  
- Fill missing numeric values with median and categorical values with mode.  
- One-hot encode categorical variables.  
- Standardize numeric features using `StandardScaler`.  
- Exclude target (`pass_fail`) from feature engineering.

---

## Modeling

Three machine learning models were trained:

1. **Logistic Regression** (with `class_weight="balanced"`)  
2. **Random Forest Classifier** (with `class_weight="balanced"`)  
3. **K-Nearest Neighbors**  

Imbalanced data was handled using **SMOTE**.

---

## Evaluation

| Model                | Accuracy | F1-Score |
|---------------------|----------|----------|
| Logistic Regression  | 0.66     | 0.70     |
| Random Forest        | 0.80     | 0.78     |
| KNN                  | 0.80     | 0.77     |

**Random Forest** was selected as the best-performing model.

---

## Limitations

- The dataset is **highly imbalanced**: far more students pass than fail.  
- Even after SMOTE, some models (like Logistic Regression and KNN) still have low recall for the failing students.  
- Accuracy can be misleading because predicting all students as passing gives high accuracy but poor detection of failing students.  
- Further improvement may require:
  - Collecting more data for the failing class  
  - Trying advanced imbalance techniques like **ADASYN**  
  - Using ensemble methods or cost-sensitive learning  

> ⚠️ **Conclusion:** This model should not be used as the sole decision tool for predicting student performance, especially for students at risk of failing.


## Usage

```
python
import joblib

# Load the trained model
model = joblib.load("model/RandomForest_model.pkl")

# Make predictions
y_pred = model.predict(X_new)

```

## Saved Model

The trained Random Forest model is saved as:

```
model/RandomForest_model.pkl
```

## Author

Muhammad Talha
