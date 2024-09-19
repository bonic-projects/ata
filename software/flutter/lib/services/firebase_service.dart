import 'package:stacked/stacked.dart';

import '../models/device.dart';
import 'package:firebase_database/firebase_database.dart';

const dbCode = "FJwEbU5AfCS5Zg8Cs2D1DfJMQuI2";

class DatabaseService with ListenableServiceMixin {
  final FirebaseDatabase _db = FirebaseDatabase.instance;

  DeviceReading? _node;
  DeviceReading? get node => _node;
  DeviceData? _node1;
  DeviceData? get node1 => _node1;

  void setupNodeListening() {
    DatabaseReference starCountRef = _db.ref('/devices/$dbCode/reading');
    DatabaseReference starCountRefer = _db.ref('/devices/$dbCode/data');

    try {
      starCountRef.onValue.listen((DatabaseEvent event) {
        print("Reading..");
        if (event.snapshot.exists) {
          print(event.snapshot.value);
          _node = DeviceReading.fromMap(event.snapshot.value as Map);
          print(_node?.lastSeen); //data['time']
          notifyListeners();
        }
      });
      starCountRefer.onValue.listen((DatabaseEvent event) {
        print("Reading l1..");
        if (event.snapshot.exists) {
          print(event.snapshot.value);
          _node1 = DeviceData.fromMap(event.snapshot.value as Map);
          print(_node1?.l1);
          print(_node1?.l2);

          notifyListeners();
        }
      });
    } catch (e) {
      print("Error: $e");
    }
  }
}
