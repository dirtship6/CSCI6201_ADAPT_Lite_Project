"""
CSCI 6201 Final Project - ADAPT Lite Fault Detection
Author: Cole VonOhlen 
Email: cole.vonohlen@gwu.edu
Date: October 21, 2026

Description: Code to pull in and process data from NASA's Advanced Diagnostics and Prognostics Testbed (ADAPT) challenge set. 
  
"""

from pathlib import Path 
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 




#Pull in data from file and folder paths using the pathlib tool and make a table
def load_data(path): 
    lines = Path(path).read_text(encoding="utf-8").splitlines()     #Reads file and splits at each row/line
    sensor_names = lines[1].split("\t")[1:]                         #Sepeates columns of the data with a \t from position 1 to the end

    sensor_names = []                                               #Creates an empty list for sensor names
    data_rows = []                                                  #Creates an empty list for rows of data
    for line in lines:                                              #Iterates through each line of the file
        parts = line.split("\t")                                    #Splits the line whenever it runs into the \t we put in
        if parts[0] == "SensorData":                                #SensorData identifies the header line in the file
                for name in parts[1:]:                              #Goes through the rest of the header to get all the sensor names
                    sensor_names.append(name.strip())               #Adds clean sensor names to our list
        elif parts[0] == "AntagonistData":                          #AntagonistData identifies a reading line
                data_rows.append(parts[1:len(sensor_names) + 1])    #Uses the the sensors names to get the 20 sensor readings and the time

    run = pd.DataFrame(data_rows, columns=sensor_names)             #Turns the data into a table as all text
   
    for name in sensor_names:                                       #Goes through all columns in the table
        if name != "Time":                                          #Enters every column except Time
            run[name] = pd.to_numeric(run[name])                    #Turns the text into real numbers 

    clock = pd.to_datetime(run["Time"].str.split(" GMT").str[0])    #Turns table values into time values
    run["TIME"] = (clock - clock[0]).dt.total_seconds()             #Subtract the first timestamp from every one to get a count in seconds
    run = run.drop(columns=["Time"])                                #Drop the old time column
    return run                                                         

#Checks the newly created table from load_data for missing raw data
def count_missing(run):
    missing_count = np.isnan(run).sum().sum                         #Looking for NaN values in the array, summing each column and all columns
    return missing_count

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
    return row

#Read answer key
def load_labels(answer_key_xls):                                    
    key = pd.read_excel(answer_key_xls, header=1)                                       #Opens the key for the run and skips the first row
    key = key.rename(columns={"File Name": "run", "Time (sec)": "Fault_Time"})          #Cleans up the column names
    key = key.dropna(subset=["run"])                                                    #Drops the bottom of the excel
    key["faulty"] = key["Component"].notna().astype(int)                                #Identifies the faulty lines
    return key

#Pulls functions together to process runs and compare to the key
def build_table(run_folder, answer_key_xls):
    labels = load_labels(answer_key_xls)                            #Takes the readings from the answer key
    
    rows = []                                                       #Creates an empty list to make one row per run
    for run_name in labels["run"]:                                  #Goes through every run in the answer key 
        run = load_data(Path(run_folder) / (run_name + ".txt"))     #Builds a file path and loads the 480 x 21 table 
        row = run_features(run)                                     #Makes a row of 60 features 
        row["run"] = run_name                                       #Labels the row name 
        rows.append(row)                                            #Adds to the list

    table = pd.DataFrame(rows)                                      #Stacks everything into one table 
    table["faulty"] = list(labels["faulty"])                        #Attaches faulty answers

    faulty_count = table["faulty"].sum()                            #Adds up the amount of faulty runs
    healthy_count = len(table) - faulty_count                       #Calculates healty runs
    print(f"Loaded {len(table)} runs: {faulty_count} faulty and {healthy_count} healthy")   #Expected output 59/27/32 and 62/32/30
    return table    

#Main block to run when called
if __name__ == "__main__":
    #Initial sample data
    train = build_table("/Users/cvo/Desktop/GW/CSCI6201_SW/Project/Data/DXC09_Industrial_Tier1SampleData_20090225", "/Users/cvo/Desktop/GW/CSCI6201_SW/Project/Data/DXC09_Industrial_Tier1SampleData_20090225/ADAPTtier1Data.xls")
    
    print(train.shape)  #Expected output (59, 62)

    #Competition data to compare to
    test = build_table("/Users/cvo/Desktop/GW/CSCI6201_SW/Project/Data/DXC09_Industrial_Tier1CompetitionData20090327", "/Users/cvo/Desktop/GW/CSCI6201_SW/Project/Data/DXC09_Industrial_Tier1CompetitionData20090327/ADAPTtier1CompetitionData.xls")

    print(test.shape)   #Expected output (62, 62)


    # Plot how many runs are healthy and faulty in each set
    train_counts = [np.sum(train["faulty"] == 0), np.sum(train["faulty"] == 1)]
    test_counts = [np.sum(test["faulty"] == 0), np.sum(test["faulty"] == 1)]
    x = np.arange(2)                    

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - 0.2, train_counts, width=0.4, label="Training (sample data)")
    bars2 = ax.bar(x + 0.2, test_counts, width=0.4, label="Test (competition data)")
    ax.bar_label(bars1)
    ax.bar_label(bars2)
    ax.set_xticks(x, ["Healthy", "Faulty"])
    ax.set(ylabel="Number of runs", title="Healthy vs. faulty runs", ylim=(0, 40))
    ax.legend()

    OUTPUT = Path("figures")
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / "healthy_vs_faulty.png", dpi=150)
