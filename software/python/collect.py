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

# Define the CSV file name and header row
csv_file = 'hydropod_data.csv'
header = ['moisture', 'temp', 'ts']

# Initialize CSV file with header if it doesn't exist
try:
    with open(csv_file, mode='x', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
except FileExistsError:
    pass

# Listen to the database changes
def on_data_change(event):
    data = event.data
    if data:
        moisture = data.get('moisture', 'N/A')
        temp = data.get('temp', 'N/A')
        ts = data.get('ts', 'N/A')
        row = [moisture, temp, ts]
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
