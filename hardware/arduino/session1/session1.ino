#include <LiquidCrystal_I2C.h> // Include the LiquidCrystal_I2C library for controlling the LCD
#include <OneWire.h> // Include the OneWire library for communication with the DS18B20 temperature sensor
#include  <DallasTemperature.h>// Include the DallasTemperature library for easier interaction with the DS18B20 sensor

LiquidCrystal_I2C lcd(0x27, 16, 2); // Initialize the LCD with I2C address 0x27 for a 16x2 display

#define moisturePin 33 // Define the pin where the moisture sensor is connected

// Temperature sensor setup
const int oneWireBus = 13; // Specify the GPIO pin where the DS18B20 temperature sensor is connected
OneWire oneWire(oneWireBus); // Create a OneWire instance to communicate with the sensor
DallasTemperature tempSensor(&oneWire); // Pass the OneWire instance to the DallasTemperature library

int moistureLevel; // Variable to store the moisture level read from the sensor
float sg; // Variable to store the calculated Specific Gravity (SG)
float apiGravity; // Variable to store the calculated API Gravity
float temperature; // Variable to store the temperature reading from the DS18B20 sensor
float density; // Variable to store the calculated density

void setup() {
  Serial.begin(115200); // Start serial communication at 115200 baud rate

  lcd.init(); // Initialize the LCD
  lcd.backlight(); // Turn on the LCD backlight
  lcd.clear(); // Clear the LCD display
  lcd.setCursor(0, 0); // Set the cursor to the first column of the first row
  lcd.print("HydroAi"); // Print "HydroAi" on the first row of the LCD
  lcd.setCursor(0, 1); // Set the cursor to the first column of the second row
  lcd.print("Starting..."); // Print "Starting..." on the second row of the LCD
  tempSensor.begin(); // Initialize the temperature sensor
  delay(2000); // Wait for 2 seconds before clearing the LCD
  lcd.clear(); // Clear the LCD display

}

void loop() {
  readMoisture(); // Read the moisture level from the sensor
  readTemperature(); // Read the temperature from the DS18B20 sensor
  calculateDensity(); // Calculate the density based on the SG
  calculateSGAndAPI(); // Calculate the SG and API Gravity based on the moisture level
  displayResults(); // Display the results on the LCD and optionally on the Serial Monitor
  delay(1000); // Wait for 1 second before updating the readings
}

void readMoisture() {
  moistureLevel = analogRead(moisturePin); // Read the analog value from the moisture sensor
}

void readTemperature() {
  tempSensor.requestTemperatures(); // Request the temperature from the DS18B20 sensor
  temperature = tempSensor.getTempCByIndex(0); // Get the temperature in Celsius from the first sensor (index 0)
}

void calculateSGAndAPI() {
  // Calculate Specific Gravity (SG)
  sg = moistureLevel / 4095.0; // Convert the moisture level to SG by normalizing it to the maximum ADC value

  // Calculate API Gravity
  apiGravity = (141.5 / sg) - 131.5; // Use the formula to calculate API Gravity from SG

}

void calculateDensity() {
  density = sg * 1000; // Calculate density by multiplying SG by 1000 (assuming water's density as 1000 kg/m³)
}

void displayResults() {
  lcd.clear(); // Clear the LCD display
  lcd.setCursor(0, 0); // Set the cursor to the first column of the first row
  lcd.print("Temp: "); // Print "Temp: " on the first row of the LCD
  lcd.print(temperature, 2); // Print the temperature value with 2 decimal points

  lcd.setCursor(0, 1); // Set the cursor to the first column of the second row
  lcd.print("Density: "); // Print "Density: " on the second row of the LCD
  lcd.print(density, 2); // Print the density value with 2 decimal points
  delay(2000); // Wait for 2 seconds to display the temperature and density
  lcd.clear(); // Clear the LCD display
  lcd.setCursor(0, 0); // Set the cursor to the first column of the first row
  lcd.print("SG: "); // Print "SG: " on the first row of the LCD
  lcd.print(sg, 2); // Print the SG value with 2 decimal points
  lcd.print(" API: "); // Print " API: " after SG on the first row of the LCD
  lcd.print(apiGravity, 2); // Print the API Gravity value with 2 decimal points
  // Optionally, print the results to the Serial Monitor for debugging
  Serial.print("Temp: "); // Print "Temp: " to the Serial Monitor
  Serial.print(temperature, 2); // Print the temperature value with 2 decimal points to the Serial Monitor
  Serial.print("°C | Density: "); // Print "°C | Density: " to the Serial Monitor
  Serial.print(density, 2); // Print the density value with 2 decimal points to the Serial Monitor
  Serial.print(" kg/m³ | SG: "); // Print " kg/m³ | SG: " to the Serial Monitor
  Serial.print(sg, 2); // Print the SG value with 2 decimal points to the Serial Monitor
  Serial.print(" | API Gravity: "); // Print " | API Gravity: " to the Serial Monitor
  Serial.println(apiGravity, 2); // Print the API Gravity value with 2 decimal points and end the line on the Serial Monitor

}