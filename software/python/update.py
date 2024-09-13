from firebase_admin import credentials, initialize_app, db
import csv
import time

# Set the credentials to access Firebase
cred = credentials.Certificate('hydroai-53e89-firebase-adminsdk-69ql5-aeb6ce7b5a.json')
initialize_app(cred, {
    'databaseURL': 'https://hydroai-53e89-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# Get a reference to the database service
ref = db.reference("/devices/FJwEbU5AfCS5Zg8Cs2D1DfJMQuI2/reading/")
data_ref = db.reference("/devices/FJwEbU5AfCS5Zg8Cs2D1DfJMQuI2/data")

# Define the CSV file name and header row
csv_file = 'hydropod_data.csv'
header = ['moisture', 'temp', 'ts', 'SG', 'API_Gravity']

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

# Listen to the database changes
def on_data_change(event):
    data = event.data
    if data:
        moisture = data.get('moisture', 'N/A')
        temp = data.get('temp', 'N/A')
        ts = data.get('ts', 'N/A')
        
        # Convert moisture value (analog output) to specific gravity and calculate API gravity
        if moisture is not None and moisture != 'N/A':
            analog_output = int(moisture)  # Ensure moisture is treated as an integer
            sg = analog_to_sg(analog_output)
            api_gravity = calculate_api_gravity(sg)

            # Format SG and API Gravity to 2 decimal points
            sg_formatted = f"{sg:.2f}"
            api_gravity_formatted = f"{api_gravity:.2f}"
        else:
            sg_formatted = 'N/A'
            api_gravity_formatted = 'N/A'

        row = [moisture, temp, ts, sg_formatted, api_gravity_formatted]
        if moisture is not None and temp is not None and ts is not None:
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
            print('Data saved to CSV:', row)

        # Update Firebase with formatted values
        data_ref.update({
            'l1': sg_formatted,
            'l2': api_gravity_formatted
        })
        print('Firebase updated with SG and API Gravity:', sg_formatted, api_gravity_formatted)

ref.listen(on_data_change)

print('Listening for data changes...')
# Keep the script running to listen for changes
while True:
    time.sleep(1)
