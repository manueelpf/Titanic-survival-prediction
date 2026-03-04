# Titanic Survival Prediction (scikit-learn)

Classic baseline ML project using the Kaggle Titanic dataset.

## Approach
- Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
- Preprocessing:
  - Median imputation + StandardScaler for numeric features
  - Most-frequent imputation + OneHotEncoder for categorical features
- Model: Logistic Regression
- Evaluation: 5-fold Stratified CV (accuracy)

## Setup
```bash
pip install -r requirements.txt