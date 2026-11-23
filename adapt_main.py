"""
CSCI 6201 Final Project - ADAPT Lite Fault Detection
Author: Cole VonOhlen 
Email: cole.vonohlen@gwu.edu
Date: November 23, 2026

Description: Main codeblock to perform overall analysis on NASA's Advanced Diagnostics and Prognostics Testbed (ADAPT) challenge set. 
  
"""

from load_data import load_set
import tree_v1
import forest_v1
import tree_v2
import forest_v2

def main():
    #Loads data from folders
    train_runs, train_labels = load_set(
        "Training",
        "Data/DXC09_Industrial_Tier1SampleData_20090225", 
        "Data/DXC09_Industrial_Tier1SampleData_20090225/ADAPTtier1Data.xls")
    test_runs, test_labels = load_set(
        "Test", 
        "Data/DXC09_Industrial_Tier1CompetitionData20090327", 
        "Data/DXC09_Industrial_Tier1CompetitionData20090327/ADAPTtier1CompetitionData.xls")

    #Runs model attempts 
    attempts = [tree_v1, forest_v1, tree_v2, forest_v2]

    results = []
    for attempt in attempts:
        result = attempt.run_attempt(train_runs, train_labels, test_runs, test_labels)
        results.append(result)

    #Prints initial chart 
    print("Attempt    Accuracy")
    for result in results:
        print(f"{result['name']:<10} {result['accuracy']: .1%} ")

    #Creates visulizations    



#Main block to run when called
if __name__ == "__main__":
    main()
    


   
