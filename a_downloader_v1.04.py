#!/usr/bin/python3
 
# Description: The function of downloader is to 
 
# Set up libraries
import time
from datetime import datetime, timedelta
import os
import json
import csv 

##############################################################
# Prepare the environment
##############################################################
# Define the directory and file paths
dir_database = "Database"
dir_php = dir_database + "/Data_PHP"
dir_csv = dir_database + "/Data_CSV"
url_data = "https://www3.septa.org/api/TrainView/index.php"
scan_count = 2 # This value represents how many times per minute it should scan for the current state of the trains.

# For use with grouper
toi_file = os.path.join(dir_database, "toi.csv")
train_line = "media"

##############################################################
# Initialize environment.
##############################################################

# Check if the directory exists, if not, create it
if not os.path.exists(dir_php):
    os.makedirs(dir_php)

if not os.path.exists(dir_csv):
    os.makedirs(dir_csv)


##############################################################
# Run
##############################################################

# Start the scheduling
while True:
    current_time_loop = datetime.now()
    next_minute = (current_time_loop + timedelta(minutes=1)).replace(second=0, microsecond=0)
    interval = 60 / scan_count

    for i in range(scan_count):
        scan_time = next_minute + timedelta(seconds=i * interval)
        time_to_sleep = (scan_time - datetime.now()).total_seconds()
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)
        current_time_loop = datetime.now()
        print(f"Scanned at {current_time_loop}")
        
        # Download the train data
        os.system("rm -r index*")
        os.system(f"wget {url_data}")
        
        # Manage the downloaded data
        # Get the current date and time in the specified format
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Define the new file name with the current date and time
        file_php = f"index_{current_time}.php"
        file_php_path = os.path.join(dir_php, file_php)
        # Move the file to the new directory with the new name
        file_index = "index.php"
        os.system(f"mv {file_index} {file_php_path}")

        # Check if the file exists
        if os.path.exists(file_php_path):
            # Open the data 
            with open(file_php_path, 'r') as file:
                train0 = file.read()

            # Parse the JSON string into a dictionary
            train1 = json.loads(train0)

            # Save this as a csv file
            file_csv = f"index_{current_time}.csv"
            file_csv_path = os.path.join(dir_csv, file_csv)

            # Open a CSV file for writing
            with open(file_csv_path, 'w', newline='') as file:
                csv_writer = csv.writer(file)
                # Write the header (keys of the dictionary)
                header = train1[0].keys()  # Assuming the JSON is a list of dictionaries
                csv_writer.writerow(header)
                # Write the rows (values of the dictionaries)
                for item in train1:
                    csv_writer.writerow(item.values())
        else:
            print(f"The file specified does not exist: {file_php_path}")
        
