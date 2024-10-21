#!/usr/bin/python3

# Description: The function of grouper is to make a csv that contains all of the trains of interest into one grouped csv.

# Set up libraries
import os
import time
import pandas as pd
from datetime import datetime

##############################################################
# Prepare the environment
##############################################################
# Define the directory and file paths
dir_database = "Database"
dir_csv = os.path.join(dir_database, "Data_CSV")
toi_file = os.path.join(dir_database, "toi.csv")
train_line = "media"

##############################################################
# Run
##############################################################

# Function to move old toi.csv file
def archive_old_toi_file():
    if os.path.exists(toi_file):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        new_name = os.path.join(dir_database, f"toi_before_{timestamp}.csv")
        os.rename(toi_file, new_name)

# Function to check for new files and process them
def process_new_files():
    current_files = set(os.listdir(dir_csv))
    new_files = current_files - processed_files

    if new_files:
        for file in new_files:
            file_path = os.path.join(dir_csv, file)
            if os.path.isfile(file_path):
                df = pd.read_csv(file_path)
                if "line" in df.columns:
                    train_line_rows = df[df["line"].str.contains(train_line, case=False, na=False)].copy()  # Use .copy() to avoid the warning
                    if not train_line_rows.empty:
                        # Extract date and time from filename
                        filename_date_time = file.split('_')[1] + ' ' + file.split('_')[2].split('.')[0]
                        date_time_obj = datetime.strptime(filename_date_time, '%Y-%m-%d %H-%M-%S')
                        
                        train_line_rows.loc[:, 'filename'] = file
                        train_line_rows.loc[:, 'date'] = date_time_obj.strftime('%Y-%m-%d')
                        train_line_rows.loc[:, 'time'] = date_time_obj.strftime('%H:%M:%S')
                        
                        if not os.path.exists(toi_file):
                            train_line_rows.to_csv(toi_file, index=False)
                        else:
                            train_line_rows.to_csv(toi_file, mode='a', header=False, index=False)

        processed_files.update(new_files)
        sort_toi_file()

# Function to sort the toi.csv file
def sort_toi_file():
    if os.path.exists(toi_file):
        df = pd.read_csv(toi_file)
        if 'filename' in df.columns and 'trainno' in df.columns:
            df = df.sort_values(by=['filename', 'trainno'])
            df.to_csv(toi_file, index=False)

# Archive old toi.csv file before starting the loop
archive_old_toi_file()

# Set to keep track of processed files
processed_files = set()

# Main loop
while True:
    process_new_files()
    time.sleep(10)
