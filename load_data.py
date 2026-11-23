"""
CSCI 6201 Final Project - ADAPT Lite Fault Detection
Author: Cole VonOhlen 
Email: cole.vonohlen@gwu.edu
Date: November 23, 2026

Description: Code to pull in and process ADAPT data to be leveraged by various models. 
  
"""

from pathlib import Path 
import pandas as pd 


#Pull in data from file and folder paths using the pathlib tool and make a table
def load_run(path): 
    lines = Path(path).read_text(encoding="utf-8").splitlines()     #Reads file and splits at each row/line

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
def check_run(run, run_name):
    assert run.shape == (480, 21), f"{run_name} has shape {run.shape}"          #
    assert run.isna().sum().sum() == 0, f"{run_name} has missing values"         #

#Read answer key
def load_labels(answer_key_xls):                                    
    key = pd.read_excel(answer_key_xls, header=1)                                       #Opens the key for the run and skips the first row
    key = key.rename(columns={"File Name": "run", "Time (sec)": "Fault_Time"})          #Cleans up the column names
    key = key.dropna(subset=["run"])                                                    #Drops the bottom of the excel
    key["faulty"] = key["Component"].notna().astype(int)                                #Identifies the faulty lines
    return key

#Pulls functions together to process runs and compare to the key
def load_set(set_name, run_folder, answer_key_xls):
    labels = load_labels(answer_key_xls)                            #Takes the readings from the answer key
    
    runs = []                                                       #Creates an empty list to make one row per run
    for run_name in labels["run"]:                                  #Goes through every run in the answer key 
        run = load_run(Path(run_folder) / (run_name + ".txt"))      #Builds a file path and loads the 480 x 21 table 
        #check_run(run, run_name)                                                   #
        runs.append(run)                                            #Adds to the list

    faulty_count = labels["faulty"].sum()                            #Adds up the amount of faulty runs
    healthy_count = len(labels) - faulty_count                       #Calculates healty runs
    print(f"Loaded {len(runs)} {set_name} runs: each with no missing values and a shape of {runs[0].shape}: {faulty_count} faulty and {healthy_count} healthy")   #Expected output 59/27/32 and 62/32/30
    return runs, labels   
