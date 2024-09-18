import torch
import torch.nn as nn
from firebase_admin import credentials, initialize_app, db
import time

# Define the same neural network model as in ml_model.py
class APIPredictor(nn.Module):
    def __init__(self):
        super(APIPredictor, self).__init__()
        self.layer1 = nn.Linear(2, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Load the model (without using scaler)
model = APIPredictor()
try:
    model.load_state_dict(torch.load('ml_model.pth'))
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# Firebase initialization
try:
    cred = credentials.Certificate('hydroai-53e89-firebase-adminsdk-69ql5-0a5599212f.json')
    initialize_app(cred, {
        'databaseURL': 'https://hydroai-53e89-default-rtdb.asia-southeast1.firebasedatabase.app/'
    })
    print("Firebase initialized successfully")
except Exception as e:
    print(f"Error initializing Firebase: {e}")

ref = db.reference("/devices/FJwEbU5AfCS5Zg8Cs2D1DfJMQuI2/reading/")
data_ref = db.reference("/devices/FJwEbU5AfCS5Zg8Cs2D1DfJMQuI2/data")

# Function to predict API gravity without using a scaler
def predict_api_gravity(temp, density):
    try:
        # Prepare input tensor (no scaling involved)
        inputs = torch.tensor([[temp, density]], dtype=torch.float32)
        print(f"Raw inputs: {inputs.numpy()}")

        with torch.no_grad():  # Disable gradient calculations for prediction
            predicted_api = model(inputs)
        
        print(f"Predicted API Gravity: {predicted_api.item()}")
        return predicted_api.item()
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# Helper functions for analog-to-sg conversion and density calculation
def analog_to_sg(analog_output, max_analog_output=4095):
    return analog_output / max_analog_output

def calculate_density(sg):
    return sg * 1000  # Density in kg/m³

# Event listener callback function
def on_data_change(event):
    print('Data change event received')
    data = event.data
    if data:
        print('Event data:', data)
        temp = data.get('temp', 'N/A')
        moisture = data.get('moisture', 'N/A') 
        
        if moisture is not None and moisture != 'N/A':
            analog_output = int(moisture)
            sg = analog_to_sg(analog_output)
            density = calculate_density(sg)
        else:
            sg = 'N/A'
            density = 'N/A'
        
        print('Temp:', temp, 'Density:', density)

        if temp != 'N/A' and density != 'N/A':
            temp = float(temp)
            density = float(density)

            predicted_api_gravity = predict_api_gravity(temp, density)
            print('Predicted API Gravity:', predicted_api_gravity)

            if predicted_api_gravity is not None:
                api_gravity_formatted = f"{predicted_api_gravity:.2f}"

                data_ref.update({
                    'l1': density,
                    'l2': api_gravity_formatted
                })
                print('Firebase updated with API Gravity:', api_gravity_formatted)
            else:
                print('Prediction failed')
        else:
            print('Data not valid:', temp, density)
    else:
        print('No data found in event')

# Set up Firebase listener
try:
    ref.listen(on_data_change)
    print('Listening for data changes...')
except Exception as e:
    print(f"Error setting up listener: {e}")

while True:
    time.sleep(1)
