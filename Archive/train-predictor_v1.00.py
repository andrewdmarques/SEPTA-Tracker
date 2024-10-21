import pandas as pd
import re

# Read the CSV file into a DataFrame
toi1 = pd.read_csv('toi.csv')

# Function to validate the rows and return cleaned dataframe and log
def clean_data(toi1):
    log = []
    valid_rows = []

    # Iterate through the rows and check conditions
    for index, row in toi1.iterrows():
        valid = True
        
        # Check if date_time is in the correct format 'yyyy-mm-dd hh:mm:ss'
        if not re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", str(row['date_time'])):
            log.append(f"Row {index} removed: Invalid date_time format {row['date_time']}")
            valid = False

        # Check if train_heading is one of 'N', 'S', 'E', 'W'
        if row['train_heading'] not in ['N', 'S', 'E', 'W']:
            log.append(f"Row {index} removed: Invalid train_heading {row['train_heading']}")
            valid = False

        # Check if poi_dist is numeric
        try:
            float(row['poi_dist'])
        except ValueError:
            log.append(f"Row {index} removed: Invalid poi_dist {row['poi_dist']}")
            valid = False

        # Check if train_id is not blank
        if pd.isna(row['train_id']) or str(row['train_id']).strip() == "":
            log.append(f"Row {index} removed: Blank train_id")
            valid = False

        # Append valid row to valid_rows list
        if valid:
            valid_rows.append(row)

    # Create a new DataFrame with valid rows
    toi2 = pd.DataFrame(valid_rows)

    # Write log to a file
    with open('log_toi.txt', 'w') as log_file:
        log_file.write("\n".join(log))

    return toi2

# Apply the function to clean the data
toi2 = clean_data(toi1)

# Save the cleaned dataframe to a new CSV file
toi2.to_csv('toi2.csv', index=False)
