import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load and prepare the dataset
data = pd.read_csv('cleaned_hydroai_data.csv')

# Extract features and target
X = data[['temp', 'Density']].values  # Features: temp and Density (without scaling)
y = data['API_Gravity'].values  # Target: API_Gravity

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Define a simple neural network model
class APIPredictor(nn.Module):
    def __init__(self):
        super(APIPredictor, self).__init__()
        self.layer1 = nn.Linear(2, 64)  # Input layer (2 features) -> Hidden layer (64 neurons)
        self.layer2 = nn.Linear(64, 32) # Hidden layer (64 neurons) -> Hidden layer (32 neurons)
        self.layer3 = nn.Linear(32, 1)  # Hidden layer (32 neurons) -> Output layer (1 value)

    def forward(self, x):
        x = torch.relu(self.layer1(x))  # Activation function for hidden layers
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)              # No activation for output layer
        return x

# Initialize the model, loss function, and optimizer
model = APIPredictor()
criterion = nn.MSELoss()  # Mean Squared Error for regression tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 3: Train the model on raw (unscaled) data
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

num_epochs = 1000  # Number of epochs (adjustable)
for epoch in range(num_epochs):
    # Forward pass: Compute predicted API_Gravity by passing inputs to the model
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()  # Clear the gradients
    loss.backward()        # Backpropagate the error
    optimizer.step()       # Update weights

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 4: Test the model (optional)
model.eval()  # Set the model to evaluation mode
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')
    
# After training, save the model (no need to save the scaler since it's not used)
torch.save(model.state_dict(), 'ml_model.pth')

# Step 5: Make predictions (using raw data)
def predict_api_gravity(temp, density):
    # Prepare input for prediction (raw data)
    inputs = torch.tensor([[temp, density]], dtype=torch.float32)
    with torch.no_grad():
        predicted_api = model(inputs)
    return predicted_api.item()

# Example prediction
temp = 30.3125
density = 699.3895
predicted_api_gravity = predict_api_gravity(temp, density)
print(f'Predicted API Gravity: {predicted_api_gravity:.2f}')
