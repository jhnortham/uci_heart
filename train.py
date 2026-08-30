"""
Model Development and Training

"""

# Import packages

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Inspect ml ready data

df = pd.read_csv("../data/processed/heart_disease_ml_ready.csv")

print(df.shape)
print(df.columns.tolist())
print(df.isna().sum())
print(df.describe())
print(df["target"].value_counts())

# Split data into training and testing sets

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=123,
    stratify=y
)

# Decide imputation method for missing values - inspect distributions

continuous_missing = [
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

print(X_train[continuous_missing].describe())
print(X_train[continuous_missing].skew())

low_chol = df.loc[df["chol"] < 120, "chol"]

print("Number below 120: ", len(low_chol))
print(low_chol.value_counts().sort_index())
print(sorted(df.loc[df["chol"] < 120, "chol"].unique()))

# Define features

continuous_features = [
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

categorical_features = [
    "restecg",
    "exang",
    "slope"
]

median_imputer = SimpleImputer(strategy="median")
mode_imputer = SimpleImputer(strategy="most_frequent")

# Fit on the training data

median_imputer.fit(X_train[continuous_features])
mode_imputer.fit(X_train[categorical_features])

print("Continuous medians: ", median_imputer.statistics_)
print("Categorical modes: ", mode_imputer.statistics_)

X_train[continuous_features] = median_imputer.transform(
    X_train[continuous_features]
)

X_test[continuous_features] = median_imputer.transform(
    X_test[continuous_features]
)

X_train[categorical_features] = mode_imputer.transform(
    X_train[categorical_features]
)

X_test[categorical_features] = mode_imputer.transform(
    X_test[categorical_features]
)

print(X_train.isna().sum())
print(X_test.isna().sum())

scale_features = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

scaler = StandardScaler()

scaler.fit(X_train[scale_features])

X_train[scale_features] = scaler.transform(
    X_train[scale_features]
)

X_test[scale_features] = scaler.transform(
    X_test[scale_features]
)

print(X_train[scale_features].mean())
print(X_train[scale_features].std())

# Fit a logistic regression model

log_model = LogisticRegression()

log_model.fit(X_train, y_train)

# Make predictions

y_pred = log_model.predict(X_test)

# View metrics

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1: ", f1_score(y_test, y_pred))
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))

    


