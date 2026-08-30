"""
Inspection of Heart dataset

"""

# Import packages and inspect data

import pandas as pd
import numpy as np

heart = pd.read_csv("../data/raw/processed.cleveland.data", header=None, na_values="?")

print(heart.shape)
print(heart.head())
print(heart.dtypes)

heart2 = pd.read_csv("../data/raw/processed.hungarian.data", header=None, na_values="?")

print(heart2.shape)
print(heart2.head())
print(heart2.dtypes)

heart3 = pd.read_csv("../data/raw/processed.switzerland.data", header=None, na_values="?")

print(heart3.shape)
print(heart3.head())
print(heart3.dtypes)

columns = [
    "age", "sex", "cp", "trespbs", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num"
]

heart.columns = columns
heart2.columns = columns
heart3.columns = columns

print(heart.isna().sum())
print(heart2.isna().sum())
print(heart3.isna().sum())

# Determine percent missingness for columns due to high number of NaNs in Hungary - ca and thal and Switzerland ca
# ca found to be important feature in cleveland-only dataset - missing in Hungary and Switzerland
# percentage supports decision-making: using all three datasets or just Cleveland?

percent_missing_heart = heart.isna().sum() * 100 / len(heart)
percent_missing_heart2 = heart2.isna().sum() * 100 / len(heart2)
percent_missing_heart3 = heart3.isna().sum() * 100/len(heart3)

print("Missingness by column for Cleveland: ", percent_missing_heart)
print("Missingness by column for Hungary: ", percent_missing_heart2)
print("Missingness by column for Switzerland: ", percent_missing_heart3)

# Inspection determined that mode for slope will be 2.0 regardless of whether imputed independently or after concatenation
print("Slope inspection for managing NaNs - Cleveland: ", heart["slope"].value_counts(dropna = False))
print("Slope inspection for managing NaNs - Hungarian: ", heart2["slope"].value_counts(dropna = False))
print("Slope inspection for managing NaNs - Switzerland: ", heart3["slope"].value_counts(dropna = False))
