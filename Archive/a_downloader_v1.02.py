#!/usr/bin/python3
 
# Set up libraries
import time
from datetime import datetime, timedelta

##############################################################
# Prepare the environment
##############################################################
# Define the directory and file paths
# Define the directory and file paths
dir_database = "Database"
dir_php = dir_database + "/Data_PHP"
dir_csv = dir_database + "/Data_CSV"


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


# Set the scan_count variable
scan_count = 6

# Start the scheduling
while True:
    current_time = datetime.now()
    next_minute = (current_time + timedelta(minutes=1)).replace(second=0, microsecond=0)
    interval = 60 / scan_count

    for i in range(scan_count):
        scan_time = next_minute + timedelta(seconds=i * interval)
        time_to_sleep = (scan_time - datetime.now()).total_seconds()
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)
        current_time = datetime.now()
        print(f"Scanned at {current_time}")
        
