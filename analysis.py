"""
CSCI 6201 Final Project - ADAPT Lite Fault Detection
Author: Cole VonOhlen 
Email: cole.vonohlen@gwu.edu
Date: November 20, 2026

Description: Code to anylize data from NASA's Advanced Diagnostics and Prognostics Testbed (ADAPT) challenge set. 
Leverage this testbed data to create a simple machine-learning model that can identify a healthy spacecraft power system from a faulty one. 
  
"""

from pathlib import Path 
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier 

from process_data import build_table

OUTPUT = Path(__file__).resolve().parent / "Figures"

train = build_table("Data/DXC09_Industrial_Tier1SampleData_20090225", "Data/DXC09_Industrial_Tier1SampleData_20090225/ADAPTtier1Data.xls")
test = build_table("Data/DXC09_Industrial_Tier1CompetitionData20090327", "Data/DXC09_Industrial_Tier1CompetitionData20090327/ADAPTtier1CompetitionData.xls")

assert train.shape == (59, 82)  # Checking data before moving on, could possibly pull from check_data
assert test.shape == (62, 82)

feature_names = list(train.columns.drop(["run", "faulty"]))
X_train = train[feature_names].to_numpy()
y_train = train["faulty"].to_numpy()
X_test = test[feature_names].to_numpy()
y_test = test["faulty"].to_numpy()

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Train healthy:", np.sum(y_train == 0 ), " faulty:", np.sum(y_train == 1))
print("Test healthy:", np.sum(y_test == 0 ), " faulty:", np.sum(y_test == 1))


model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
matrix = confusion_matrix(y_test, predictions)
print(f"Test accuracy: {accuracy:.1%}")


from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(random_state=42)
forest.fit(X_train, y_train)
forest_predictions = forest.predict(X_test)
print(f"Forest accuracy: {accuracy_score(y_test, forest_predictions): .1%}")