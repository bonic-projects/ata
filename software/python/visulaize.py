import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned data
session1data = pd.read_csv('cleaned_actual_data.csv')
session2data = pd.read_csv('cleaned_predicted_data.csv')

# Extracting relevant columns
hydrometer_session1 = session1data['moisture']
actual_api_gravity = session1data['API_Gravity']

hydrometer_session2 = session2data['hydrometer']
predicted_api_gravity = session2data['API_Gravity']  # Model-predicted values for session two

# Plotting the data
plt.figure(figsize=(10, 6))

# Session One: Actual API gravity vs. Hydrometer sensor reading
plt.subplot(2, 1, 1)  # Create a 2-row, 1-column layout, this is the first plot
plt.plot(hydrometer_session1, actual_api_gravity, label="Actual API Gravity", color='blue')
plt.title('Session One: Actual API Gravity vs. Hydrometer')
plt.xlabel('Hydrometer Sensor Reading')
plt.ylabel('Actual API Gravity')
plt.grid(True)
plt.legend()

# Session Two: Predicted API gravity vs. Hydrometer sensor reading
plt.subplot(2, 1, 2)  # This is the second plot
plt.plot(hydrometer_session2, predicted_api_gravity, label="Predicted API Gravity", color='red')
plt.title('Session Two: Predicted API Gravity vs. Hydrometer Sensor (ML Model)')
plt.xlabel('Hydrometer Sensor Reading')
plt.ylabel('Predicted API Gravity')
plt.grid(True)
plt.legend()

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()
