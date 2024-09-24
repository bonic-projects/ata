from firebase_admin import credentials, initialize_app, db

def init():
    # Set the credentials to access Firebase
    
    try:
        cred = credentials.Certificate('config.json')
        initialize_app(cred, {
            'databaseURL': 'https://hydroai-53e89-default-rtdb.asia-southeast1.firebasedatabase.app/'
        })
        print("Bonic Cloud initialized successfully")
    except Exception as e:
        print(f"Bonic Cloud initializing Error: {e}")


def get_ref():
    return db.reference("/devices/FJwEbU5AfCS5Zg8Cs2D1DfJMQuI2/reading/")


def get_data_ref():
    return db.reference("/devices/FJwEbU5AfCS5Zg8Cs2D1DfJMQuI2/data")