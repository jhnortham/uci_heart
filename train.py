"""
Model Development and Training

"""

# Import packages

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

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

# Create copies of raw train and test datasets for final model
X_train_raw = X_train.copy()
X_test_raw = X_test.copy()

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

# Inspect correlations between features and targets for selecting important features

train_analysis = X_train.copy()
train_analysis["target"] = y_train.to_numpy()

target_correlations = (
    train_analysis.corr()["target"].sort_values(ascending=False)
)

print(target_correlations)


# Scale features
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

# Feature selection based on training correlations
# Removal of weakest column: restecg

X_train_selected = X_train.drop(columns=["restecg"])
X_test_selected = X_test.drop(columns=["restecg"])

print(X_train_selected.columns)
print(X_train_selected.shape)

# Create Logistic Regression with removed column

lr_selected = LogisticRegression()

lr_selected.fit(X_train_selected, y_train)

y_pred_selected = lr_selected.predict(X_test_selected)

# Calculate metrics

print("Accuracy: ", accuracy_score(y_test, y_pred_selected))
print("Precision: ", precision_score(y_test, y_pred_selected))
print("Recall: ", recall_score(y_test, y_pred_selected))
print("F1: ", f1_score(y_test, y_pred_selected))
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred_selected))

# Remove two weakest columns: restecg and chol

X_train_selected2 = X_train_selected.drop(columns=["chol"])
X_test_selected2 = X_test_selected.drop(columns=["chol"])

print(X_train_selected2.columns)
print(X_train_selected2.shape)

# Create Logistic Regression with additional removed column

lr_selected2 = LogisticRegression()

lr_selected2.fit(X_train_selected2, y_train)

y_pred_selected2 = lr_selected2.predict(X_test_selected2)

# Calculate metrics

print("Accuracy: ", accuracy_score(y_test, y_pred_selected2))
print("Precision: ", precision_score(y_test, y_pred_selected2))
print("Recall: ", recall_score(y_test, y_pred_selected2))
print("F1: ", f1_score(y_test, y_pred_selected2))
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred_selected2))

# Final model for deployment

selected_features = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "thalach",
    "exang",
    "oldpeak",
    "slope"
]

continuous_model_features = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

categorical_model_features = [
    "sex",
    "cp",
    "exang",
    "slope"
]


X_train_final = X_train_raw[selected_features]
X_test_final = X_test_raw[selected_features]

print(X_train_final.shape)
print(X_test_final.shape)
print(X_train_final.isna().sum())

# Build continuous pipeline

continuous_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Build categorical pipeline

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

# Combine columns

preprocessor = ColumnTransformer([
    ("continuous", continuous_pipeline, continuous_model_features),
    ("categorical", categorical_pipeline, categorical_model_features)
])

# Combine preprocessor with logistic regression model (model #2)

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

final_pipeline.fit(X_train_final, y_train)

y_pred_final = final_pipeline.predict(X_test_final)


# Calculate metrics for final pipeline

print("Accuracy: ", accuracy_score(y_test, y_pred_final))
print("Precision: ", precision_score(y_test, y_pred_final))
print("Recall: ", recall_score(y_test, y_pred_final))
print("F1: ", f1_score(y_test, y_pred_final))
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred_final))

# Serialization of model for deployment

joblib.dump(final_pipeline, "../models/heart_disease_pipeline.joblib")


    


