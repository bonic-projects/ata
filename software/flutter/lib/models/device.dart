class DeviceReading {
  double temp;
  double moisture;
  DateTime lastSeen;

  DeviceReading({
    required this.temp,
    required this.moisture,
    required this.lastSeen,
  });

  factory DeviceReading.fromMap(Map data) {
    return DeviceReading(
      temp: data['temp'] != null
          ? (data['temp'] % 1 == 0 ? data['temp'] + 0.1 : data['temp'])
          : 0.0,
      lastSeen: DateTime.fromMillisecondsSinceEpoch(data['ts']),
      moisture: data['moisture'] != null
          ? (data['moisture'] % 1 == 0
              ? data['moisture'] + 0.1
              : data['moisture'])
          : 0.0,
    );
  }
}

class DeviceData {
  String l1;
  String l2;

  DeviceData({
    required this.l1,
    required this.l2,
  });

  // Factory constructor to create DeviceData from a JSON map
  factory DeviceData.fromMap(Map data) {
    return DeviceData(
      l1: data['l1'] ?? '', // Default to empty string if l1 is null
      l2: data['l2'] ?? '', // Default to empty string if l2 is null
    );
  }

  // Method to convert DeviceData to JSON
  Map<String, dynamic> toJson() {
    return {
      'l1': l1,
      'l2': l2,
    };
  }
}
