#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);  // Set the LCD address to 0x27 for a 16 chars and 2 line display
#define moisturePin 33

//Temperature sensor
#include <OneWire.h>
#include <DallasTemperature.h>
// // GPIO where the DS18B20 is connected to
const int oneWireBus = 13;          
// Setup a oneWire instance to communicate with any OneWire devices
OneWire oneWire(oneWireBus);
// Pass our oneWire reference to Dallas Temperature sensor
DallasTemperature tempSensor(&oneWire);
// // Temperature value
float temperature;
int moistureLevel;

//WiFi
#define wifiLedPin 5

//Firebase
#include <Arduino.h>
#include <WiFi.h>
#include <FirebaseESP32.h>
// Provide the token generation process info.
#include <addons/TokenHelper.h>
// Provide the RTDB payload printing info and other helper functions.
#include <addons/RTDBHelper.h>
/* 1. Define the WiFi credentials */
#define WIFI_SSID "Autobonics_4G"
#define WIFI_PASSWORD "autobonics@27"
// For the following credentials, see examples/Authentications/SignInAsUser/EmailPassword/EmailPassword.ino
/* 2. Define the API Key */
#define API_KEY "AIzaSyBzWb2JnuP7K05Qjc3okhCCt9j8t9pae5w"
/* 3. Define the RTDB URL */
#define DATABASE_URL "https://hydroai-53e89-default-rtdb.asia-southeast1.firebasedatabase.app/" //<databaseName>.firebaseio.com or <databaseName>.<region>.firebasedatabase.app
/* 4. Define the user Email and password that alreadey registerd or added in your project */
#define USER_EMAIL "device@gmail.com"
#define USER_PASSWORD "12345678"
// Define Firebase Data object
FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;
unsigned long sendDataPrevMillis = 0;
// Variable to save USER UID
String uid;
//Databse
String path;

String line1 = "";
String line2 = "";


FirebaseData stream;
void streamCallback(StreamData data)
{
  Serial.println("NEW DATA!");

  String p = data.dataPath();

  Serial.println(p);
  printResult(data); // see addons/RTDBHelper.h

  FirebaseJson jVal = data.jsonObject();
  FirebaseJsonData line1FB;
  FirebaseJsonData line2FB;

  jVal.get(line1FB, "l1");
  jVal.get(line2FB, "l2");

   if (line1FB.success)
  {
    Serial.println("Success data line1FB");
    String value = line1FB.to<String>(); 
    line1 = value;
    lcd.clear(); 
    lcd.setCursor(0, 0);
    lcd.print("SG:");
    lcd.print(line1);
    lcd.print("API Gravity:");
    lcd.setCursor(0, 1);
    lcd.print(line2);
  }
   if (line2FB.success)
  {
    Serial.println("Success data line2FB");
    String value = line2FB.to<String>(); 
    line2 = value;
    lcd.clear(); 
    lcd.setCursor(0, 0);
    lcd.print("Density:");
    lcd.print(line1);
    lcd.setCursor(0, 1);
    lcd.print("API Gravity:");
    lcd.print(line2);

  }
}

void streamTimeoutCallback(bool timeout)
{
  if (timeout)
    Serial.println("stream timed out, resuming...\n");

  if (!stream.httpConnected())
    Serial.printf("error code: %d, reason: %s\n\n", stream.httpCode(), stream.errorReason().c_str());
}

unsigned long printDataPrevMillis = 0;


void setup() {

  Serial.begin(115200);

  lcd.init();
  lcd.clear();         
  lcd.backlight(); 

  //Display
  lcd.init();
  lcd.backlight();
  lcd.clear(); 
  lcd.setCursor(0, 0);
  lcd.print("HydroAi");
  lcd.setCursor(0, 1);
  lcd.print("Connecting wifi..");

  //Temperature
  tempSensor.begin();
 
 
  //WIFI
  pinMode(wifiLedPin, OUTPUT);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to Wi-Fi");
  unsigned long ms = millis();
  while (WiFi.status() != WL_CONNECTED)
  {
    digitalWrite(wifiLedPin, LOW);
    Serial.print(".");
    delay(300);
  }
  Serial.println();
  Serial.print("Connected with IP: ");
  digitalWrite(wifiLedPin, HIGH);
  Serial.println(WiFi.localIP());
  Serial.println();

  //FIREBASE
  Serial.printf("Firebase Client v%s\n\n", FIREBASE_CLIENT_VERSION);
  /* Assign the api key (required) */
  config.api_key = API_KEY;

  /* Assign the user sign in credentials */
  auth.user.email = USER_EMAIL;
  auth.user.password = USER_PASSWORD;

  /* Assign the RTDB URL (required) */
  config.database_url = DATABASE_URL;

  /* Assign the callback function for the long running token generation task */
  config.token_status_callback = tokenStatusCallback; // see addons/TokenHelper.h

  // Limit the size of response payload to be collected in FirebaseData
  fbdo.setResponseSize(2048);

  Firebase.begin(&config, &auth);

  // Comment or pass false value when WiFi reconnection will control by your code or third party library
  Firebase.reconnectWiFi(true);

  Firebase.setDoubleDigits(5);

  config.timeout.serverResponse = 10 * 1000;

  // Getting the user UID might take a few seconds
  Serial.println("Getting User UID");
  while ((auth.token.uid) == "") {
    Serial.print('.');
    delay(1000);
  }
  // Print user UID
  uid = auth.token.uid.c_str();
  Serial.print("User UID: ");
  Serial.println(uid);

  path = "devices/" + uid + "/reading";
    lcd.clear(); 
  lcd.setCursor(0, 0);
  lcd.print("HydroAI");

     //Stream setup
  if (!Firebase.beginStream(stream, "devices/" + uid + "/data"))
    Serial.printf("sream begin error, %s\n\n", stream.errorReason().c_str());

  Firebase.setStreamCallback(stream, streamCallback, streamTimeoutCallback);
}

void loop() {
    readMoisture();
    readTemp();
    printData();
    updateData();
}

void updateData(){
  if (Firebase.ready() && (millis() - sendDataPrevMillis > 2000 || sendDataPrevMillis == 0))
  {
    sendDataPrevMillis = millis();
    FirebaseJson json;
    json.set("moisture", moistureLevel);
    json.set("temp", temperature);
    json.set(F("ts/.sv"), F("timestamp"));
    Serial.printf("Set data with timestamp... %s\n", Firebase.setJSON(fbdo, path.c_str(), json) ? fbdo.to<FirebaseJson>().raw() : fbdo.errorReason().c_str());
    Serial.println(""); 
  }
}

void printData(){
  if (millis() - printDataPrevMillis > 2000 || printDataPrevMillis == 0)
  {
    printDataPrevMillis = millis();
    //PH
    Serial.print("Moisture: ");
    Serial.print(moistureLevel);

    //Temperature
    Serial.print("Temperature:");
    Serial.print(temperature, 2);
    Serial.println("ºC");
  }
}

void readMoisture(){
 moistureLevel = analogRead(moisturePin);
} 

void readTemp(){
  tempSensor.requestTemperatures();
  temperature = tempSensor.getTempCByIndex(0);

}
