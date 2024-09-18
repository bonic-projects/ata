import csv

# Input and output CSV file names
input_file = 'hydroai_data.csv'
output_file = 'cleaned_hydroai_data.csv'

# Define the columns to extract in the desired order
columns_to_extract = ['temp', 'Density', 'API_Gravity']

# Open the input CSV file for reading
with open(input_file, mode='r', newline='') as infile:
    reader = csv.DictReader(infile)

    # Open the output CSV file for writing
    with open(output_file, mode='w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=columns_to_extract)

        # Write the header row in the new CSV
        writer.writeheader()

        # Write only the required columns to the new CSV file
        for row in reader:
            cleaned_row = {col: row[col] for col in columns_to_extract}
            writer.writerow(cleaned_row)

print(f"Data cleaned and written to {output_file}")
