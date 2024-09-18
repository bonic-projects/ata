from firebase_admin import credentials, initialize_app, db
import csv
import time

# Set the credentials to access Firebase
cred = credentials.Certificate('hydroai-53e89-firebase-adminsdk-69ql5-0a5599212f.json')
initialize_app(cred, {
    'databaseURL': 'https://hydroai-53e89-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# Get a reference to the database service
ref = db.reference("/devices/FJwEbU5AfCS5Zg8Cs2D1DfJMQuI2/reading/")

# Define the CSV file name and header row
csv_file = 'hydroai_data.csv'
header = ['moisture', 'temp', 'ts', 'SG', 'API_Gravity', 'Density']

# Initialize CSV file with header if it doesn't exist
try:
    with open(csv_file, mode='x', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
except FileExistsError:
    pass

# Convert Analog Output to Specific Gravity
def analog_to_sg(analog_output, max_analog_output=4095):
    return analog_output / max_analog_output

# Calculate API Gravity from Specific Gravity
def calculate_api_gravity(sg):
    return (141.5 / sg) - 131.5

# Calculate Density from Specific Gravity (SG)
def calculate_density(sg):
    return sg * 1000  # Density in kg/m³

# Listen to the database changes
def on_data_change(event):
    data = event.data
    if data:
        moisture = data.get('moisture', 'N/A')
        temp = data.get('temp', 'N/A')
        ts = data.get('ts', 'N/A')
        
        # Convert moisture value (analog output) to specific gravity, calculate API gravity and density
        if moisture is not None and moisture != 'N/A':
            analog_output = int(moisture)  # Ensure moisture is treated as an integer
            sg = analog_to_sg(analog_output)
            api_gravity = calculate_api_gravity(sg)
            density = calculate_density(sg)
        else:
            sg = 'N/A'
            api_gravity = 'N/A'
            density = 'N/A'

        # Add the calculated data to the row
        row = [moisture, temp, ts, sg, api_gravity, density]

        # Save the row to the CSV file
        if moisture is not None and temp is not None and ts is not None:
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
            print('Data saved to CSV:', row)

ref.listen(on_data_change)

print('Listening for data changes...')
# Keep the script running to listen for changes
while True:
    time.sleep(1)
