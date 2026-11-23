"""
CSCI 6201 Final Project - ADAPT Lite Fault Detection
Author: Cole VonOhlen 
Email: cole.vonohlen@gwu.edu
Date: November 23, 2026

Description: Code to pull in and process data from NASA's Advanced Diagnostics and Prognostics Testbed (ADAPT) challenge set. 
  
"""

import numpy as np 
import pandas as pd 
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier 


#Take a full run table and condense it into one line of 60 features (3 per sensor)
def run_features(run):
    start = run[run["TIME"] <= 30]                                  #No faults were in the inital 30 seconds
    window = run[run["TIME"] > 30]                                  #Test window went form 30 to the end of the test

    row = {}                                                        #Creates an empty dictionary to hold the values
    for sensor in run.columns:                                      #Iterates through each sensor
        if sensor != "TIME":                                        #Visits all 20 sensor columns but skips TIME
            row[sensor + "_mean_change"] = window[sensor].mean() - start[sensor].mean() #Windos average minus start average
            row[sensor + "_spread_change"] = window[sensor].std() - start[sensor].std() #How much the readings shift after the start versus before
            row[sensor + "_slope"] = np.polyfit(window["TIME"], window[sensor], 1)[0]   #Fits a line through the windows readings
            row[sensor + "max_jump"] = (window[sensor] - start[sensor].mean()).abs().max()  #
    return row


#Pulls functions together to process runs and compare to the key
def build_table(runs):
    rows = []                                                       #Creates an empty list to make one row per run
    for run in runs:                                  #Goes through every run in the answer key 
        rows.append(run_features(run))                                            #Adds to the list
    return pd.DataFrame(rows)


def run_attempt(train_runs, train_labels, test_runs, test_labels):
    X_train = build_table(train_runs)
    X_test = build_table(test_runs)
    y_train = train_labels["faulty"].to_numpy()
    y_test = test_labels["faulty"].to_numpy()

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        "name": "tree_v2",
        "accuracy": accuracy_score(y_test, predictions),
        "correct": int(np.sum(predictions == y_test)),
        "total": len(y_test),
        "matrix": confusion_matrix(y_test, predictions),
        "importance": pd.Series(model.feature_importances_, index=X_train.columns),
    }
 


