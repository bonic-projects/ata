import 'package:hydro_ai/app/app.locator.dart';
import 'package:hydro_ai/services/firebase_service.dart';
import 'package:stacked/stacked.dart';

import '../../../models/device.dart';

class HomeViewModel extends ReactiveViewModel {
  final _databaseService = locator<DatabaseService>();

  DeviceReading? get node => _databaseService.node;
  DeviceData? get node1 => _databaseService.node1;

  @override
  List<ListenableServiceMixin> get listenableServices => [_databaseService];
  Future runStartupLogic() async {
    _databaseService.setupNodeListening();
  }

  // Apply moisture conversion: (moisture / 4095) * 100
  double get moisture => (node?.moisture ?? 0) / 4095 * 100;

  double get temperature => node?.temp ?? 0.0;

  // Ensure l1 and l2 are always returned as strings
  String get l1 => node1?.l1.toString() ?? 'N/A';
  String get l2 => node1?.l2.toString() ?? 'N/A';
}
