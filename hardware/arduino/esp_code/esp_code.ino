#include <LiquidCrystal_I2C.h>

// Initialize the LCD display with I2C address 0x27 and 16 characters on 2 lines
LiquidCrystal_I2C lcd(0x27, 16, 2);  

#define moisturePin 33  // Define the pin for the moisture sensor

// Temperature sensor
#include <OneWire.h> // Library for OneWire communication (used by temperature sensors)
#include <DallasTemperature.h> // Library for Dallas temperature sensors (DS18B20, etc.)

// Pin where the DS18B20 temperature sensor is connected
const int oneWireBus = 13;          

// Create an instance of the OneWire library for communication with the DS18B20
OneWire oneWire(oneWireBus);

// Create an instance of DallasTemperature library using the OneWire object
DallasTemperature tempSensor(&oneWire);

// Variable to store temperature reading
float temperature;
// Variable to store moisture level reading
int moistureLevel;

// WiFi setup
#define wifiLedPin 5  // Pin connected to the LED that indicates WiFi status

// Firebase libraries
#include <Arduino.h>
#include <WiFi.h>
#include <FirebaseESP32.h>
#include <addons/TokenHelper.h> // Helps with token generation for Firebase
#include <addons/RTDBHelper.h>  // Helps with printing Firebase Real-Time Database payloads

/* 1. WiFi credentials */
#define WIFI_SSID "Autobonics_4G"
#define WIFI_PASSWORD "autobonics@27"

/* 2. Firebase API Key */
#define API_KEY "AIzaSyBzWb2JnuP7K05Qjc3okhCCt9j8t9pae5w"

/* 3. Firebase Realtime Database URL */
#define DATABASE_URL "https://hydroai-53e89-default-rtdb.asia-southeast1.firebasedatabase.app/"

/* 4. Firebase User Email and Password */
#define USER_EMAIL "device@gmail.com"
#define USER_PASSWORD "12345678"

// Create Firebase Data objects for handling data and authentication
FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;
unsigned long sendDataPrevMillis = 0; // Timestamp for data updates
String uid; // User ID (UID) to identify the device in Firebase
String path; // Firebase path to store sensor data

// Variables for the LCD display content (lines 1 and 2)
String line1 = "";
String line2 = "";

// Create Firebase Data object for streaming data
FirebaseData stream;

/* Callback function to handle data when there is a change in the stream (Realtime Database) */
void streamCallback(StreamData data)
{
  Serial.println("NEW DATA!");

  String p = data.dataPath(); // Get the path of the data that changed
  Serial.println(p);
  printResult(data); // Function to print the stream data (defined in RTDBHelper.h)

  // Create FirebaseJson objects to parse the incoming data
  FirebaseJson jVal = data.jsonObject();
  FirebaseJsonData line1FB; // Data object for the first line
  FirebaseJsonData line2FB; // Data object for the second line

  // Get the values from the Firebase data and update the LCD display
  jVal.get(line1FB, "l1");
  jVal.get(line2FB, "l2");

  if (line1FB.success) // Check if the first line data was successfully retrieved
  {
    Serial.println("Success data line1FB");
    String value = line1FB.to<String>();  // Convert the data to a string
    line1 = value; // Store the string in the line1 variable
    lcd.clear();  // Clear the display
    lcd.setCursor(0, 0);  // Set cursor to the first row
    lcd.print("SG:"); // Print "SG:" (Specific Gravity)
    lcd.print(line1); // Print the SG value
    lcd.print("API Gravity:"); // Print "API Gravity:"
    lcd.setCursor(0, 1);  // Move cursor to the second row
    lcd.print(line2); // Print the second line (API Gravity)
  }

  if (line2FB.success) // Check if the second line data was successfully retrieved
  {
    Serial.println("Success data line2FB");
    String value = line2FB.to<String>(); // Convert the data to a string
    line2 = value; // Store the string in the line2 variable
    lcd.clear(); // Clear the display
    lcd.setCursor(0, 0); // Set cursor to the first row
    lcd.print("Density:"); // Print "Density:"
    lcd.print(line1); // Print the density value
    lcd.setCursor(0, 1); // Move cursor to the second row
    lcd.print("API Gravity:"); // Print "API Gravity:"
    lcd.print(line2); // Print the second line (API Gravity)
  }
}

// Callback function for stream timeout
void streamTimeoutCallback(bool timeout)
{
  if (timeout)
    Serial.println("Stream timed out, resuming...");

  if (!stream.httpConnected()) // Check if the connection to Firebase is lost
    Serial.printf("Error code: %d, reason: %s\n", stream.httpCode(), stream.errorReason().c_str());
}

unsigned long printDataPrevMillis = 0; // Timestamp for printing sensor data

void setup() {
  Serial.begin(115200); // Begin serial communication at 115200 baud rate

  // Initialize the LCD
  lcd.init();
  lcd.clear(); // Clear the LCD screen
  lcd.backlight(); // Turn on the backlight

  // Display initial message on LCD
  lcd.setCursor(0, 0); 
  lcd.print("HydroAi"); // Print "HydroAi"
  lcd.setCursor(0, 1);
  lcd.print("Connecting wifi.."); // Print "Connecting wifi.."

  // Initialize the temperature sensor
  tempSensor.begin();

  // Initialize WiFi and attempt to connect
  pinMode(wifiLedPin, OUTPUT); // Set WiFi LED pin as output
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD); // Connect to WiFi
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) // Wait until connected
  {
    digitalWrite(wifiLedPin, LOW); // Keep LED off while connecting
    Serial.print("."); // Print dots to indicate connection attempt
    delay(300); // Small delay before retry
  }
  Serial.println(); // Newline after connection
  Serial.print("Connected with IP: ");
  digitalWrite(wifiLedPin, HIGH); // Turn LED on after successful connection
  Serial.println(WiFi.localIP()); // Print the local IP address

  // Firebase Client initialization
  Serial.printf("Firebase Client v%s\n", FIREBASE_CLIENT_VERSION);
  config.api_key = API_KEY; // Set the Firebase API key
  auth.user.email = USER_EMAIL; // Set the user email
  auth.user.password = USER_PASSWORD; // Set the user password
  config.database_url = DATABASE_URL; // Set the Firebase Realtime Database URL
  config.token_status_callback = tokenStatusCallback; // Handle token status updates

  Firebase.begin(&config, &auth); // Begin Firebase connection

  // Enable automatic WiFi reconnection
  Firebase.reconnectWiFi(true);

  // Set Firebase timeout for server responses to 10 seconds
  config.timeout.serverResponse = 10 * 1000;

  // Wait for Firebase to provide the user UID (might take a few seconds)
  Serial.println("Getting User UID");
  while (auth.token.uid == "") {
    Serial.print('.'); // Print dots while waiting
    delay(1000); // Delay before checking again
  }
  uid = auth.token.uid.c_str(); // Store the UID
  Serial.print("User UID: ");
  Serial.println(uid); // Print the UID

  path = "devices/" + uid + "/reading"; // Define the Firebase path for sensor readings

  // Setup Firebase stream to listen for changes in the "data" node
  if (!Firebase.beginStream(stream, "devices/" + uid + "/data"))
    Serial.printf("Stream begin error: %s\n", stream.errorReason().c_str());

  // Set up stream callbacks for handling data and timeouts
  Firebase.setStreamCallback(stream, streamCallback, streamTimeoutCallback);
}

void loop() {
  readMoisture(); // Read the moisture level
  readTemp(); // Read the temperature
  printData(); // Print sensor data to the serial monitor
  updateData(); // Update Firebase with new sensor data
}

// Function to send data to Firebase
void updateData() {
  if (Firebase.ready() && (millis() - sendDataPrevMillis > 2000 || sendDataPrevMillis == 0)) {
    sendDataPrevMillis = millis(); // Update the timestamp
    FirebaseJson json; // Create a Firebase JSON object
    json.set("moisture", moistureLevel); // Add moisture data to JSON
    json.set("temp", temperature); // Add temperature data to JSON
    json.set(F("ts/.sv"), F("timestamp")); // Add server timestamp to JSON
    Serial.printf("Set data with timestamp... %s\n", Firebase.setJSON(fbdo, path.c_str(), json) ? fbdo.to<FirebaseJson>().raw() : fbdo.errorReason().c_str());
    Serial.println(); // Newline for formatting
  }
}

// Function to print sensor data to the serial monitor
void printData() {
  if (millis() - printDataPrevMillis > 2000 || printDataPrevMillis == 0) {
    printDataPrevMillis = millis(); // Update the timestamp
    Serial.print("Moisture Level: ");
    Serial.println(moistureLevel); // Print moisture level
    Serial.print("Temperature: ");
    Serial.println(temperature); // Print temperature
  }
}

// Function to read moisture level from the sensor
void readMoisture() {
  moistureLevel = analogRead(moisturePin); // Read analog value from the moisture sensor pin
}

// Function to read temperature from the DS18B20 sensor
void readTemp() {
  tempSensor.requestTemperatures(); // Request temperature from sensor
  temperature = tempSensor.getTempCByIndex(0); // Get temperature in Celsius from sensor
}
