import 'package:hydro_ai/app/app.locator.dart';
import 'package:hydro_ai/models/device.dart';
import 'package:hydro_ai/services/firebase_service.dart';
import 'package:stacked/stacked.dart';

class HomeViewModel extends ReactiveViewModel {
  final _databaseService = locator<DatabaseService>();
  double _moisture = 0.0; // Example default value
  double _temperature = 0.0;
  DeviceReading? get node => _databaseService.node;
  DeviceData? get node1 => _databaseService.node1;

  @override
  List<ListenableServiceMixin> get listenableServices =>
      [_databaseService]; // Example default value

  double get moisture => node?.moisture ?? 0;
  double get temperature => node?.temp ?? 0.0;
}
