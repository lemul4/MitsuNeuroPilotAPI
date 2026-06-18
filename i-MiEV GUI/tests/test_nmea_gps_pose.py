import time
import unittest

import main
from hardware.pose_provider import Nmea0183Parser, NmeaFix, NmeaFixFilter
from vehicle_control.adapters import MockVehicleAdapter, VehicleAdapterFactory
from vehicle_control.arbiter import ControlArbiter
from vehicle_control.control_service import VehicleControlService
from vehicle_control.models import Pose2D


class Nmea0183ParserTests(unittest.TestCase):
    def test_parses_valid_gga_and_rmc(self):
        parser = Nmea0183Parser()

        gga = parser.parse("$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47")
        rmc = parser.parse("$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A")

        self.assertIsNotNone(gga)
        self.assertTrue(gga.valid)
        self.assertEqual(gga.satellites, 8)
        self.assertAlmostEqual(gga.hdop, 0.9)
        self.assertAlmostEqual(gga.lat, 48.1173, places=4)
        self.assertAlmostEqual(gga.lon, 11.5166667, places=4)
        self.assertIsNotNone(rmc)
        self.assertTrue(rmc.valid)
        self.assertAlmostEqual(rmc.course_deg, 84.4)
        self.assertAlmostEqual(rmc.speed_mps, 22.4 * 0.514444, places=4)

    def test_parses_real_gll_and_rmc_coordinates_as_degrees_minutes(self):
        parser = Nmea0183Parser()

        gll = parser.parse("$GNGLL,5559.69767,N,09247.92921,E,062555.00,A,A*7A")
        rmc = parser.parse("$GNRMC,062556.00,A,5559.69773,N,09247.92917,E,0.029,,180626,,,A*60")

        self.assertIsNotNone(gll)
        self.assertTrue(gll.valid)
        self.assertAlmostEqual(gll.lat, 55.9949611667, places=7)
        self.assertAlmostEqual(gll.lon, 92.7988201667, places=7)
        self.assertIsNotNone(rmc)
        self.assertTrue(rmc.valid)
        self.assertAlmostEqual(rmc.lat, 55.9949621667, places=7)
        self.assertAlmostEqual(rmc.lon, 92.7988195, places=7)

    def test_rejects_sentence_with_bad_checksum(self):
        parser = Nmea0183Parser()
        self.assertIsNone(
            parser.parse("$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*00")
        )


class NmeaFixFilterTests(unittest.TestCase):
    def test_requires_stable_lock_and_rejects_city_jump(self):
        filter_ = NmeaFixFilter(
            min_satellites=6,
            max_hdop=2.0,
            lock_samples=3,
            lock_radius_m=10.0,
            max_jump_m=30.0,
            window_size=3,
        )
        base = dict(valid=True, fix_quality=1, satellites=10, hdop=0.8)

        self.assertIsNone(filter_.accept(NmeaFix(56.0100000, 92.8700000, **base)))
        self.assertIsNone(filter_.accept(NmeaFix(56.0100100, 92.8700100, **base)))
        accepted = filter_.accept(NmeaFix(56.0100200, 92.8700200, **base))
        self.assertIsNotNone(accepted)

        jump = filter_.accept(NmeaFix(55.7558, 37.6173, **base))
        self.assertIsNone(jump)

    def test_rejects_low_quality_fix(self):
        filter_ = NmeaFixFilter(min_satellites=6, max_hdop=2.0, lock_samples=1)
        self.assertIsNone(filter_.accept(NmeaFix(56.0, 92.0, True, 1, 3, 0.8)))
        self.assertIsNone(filter_.accept(NmeaFix(56.0, 92.0, True, 1, 10, 8.0)))
        self.assertIsNone(filter_.accept(NmeaFix(56.0, 92.0, False, 0, 10, 0.8)))

    def test_accepts_rmc_gll_only_quality_unknown(self):
        filter_ = NmeaFixFilter(lock_samples=1)
        accepted = filter_.accept(NmeaFix(55.9949621667, 92.7988195, True, 1, 0, 99.0))
        self.assertIsNotNone(accepted)


class PoseFreshnessTests(unittest.TestCase):
    def test_nmea_lat_lon_is_converted_to_route_local_meters(self):
        mock = MockVehicleAdapter()
        service = VehicleControlService(
            VehicleAdapterFactory(real_adapter=None, mock_adapter=mock),
            ControlArbiter(),
        )
        service.adapter = mock
        controller = main.AppController.__new__(main.AppController)
        controller.vehicle_control = service
        controller.real_mission = {
            "metadata": {"origin_geo": {"lat": 56.0, "lon": 92.0}}
        }

        controller.handle_real_pose_payload({
            "lat": 56.0001,
            "lon": 92.0001,
            "yaw_deg": 15.0,
            "valid": True,
            "source": "nmea0183_gnss",
        })

        telemetry = service.get_telemetry()
        self.assertTrue(telemetry.pose_valid)
        self.assertEqual(telemetry.pose_source, "nmea0183_gnss")
        self.assertGreater(telemetry.x_m, 5.0)
        self.assertGreater(telemetry.y_m, 10.0)
        self.assertAlmostEqual(telemetry.yaw_deg, 15.0)

    def test_nmea_fix_from_another_city_is_not_accepted(self):
        mock = MockVehicleAdapter()
        service = VehicleControlService(
            VehicleAdapterFactory(real_adapter=None, mock_adapter=mock),
            ControlArbiter(),
        )
        service.adapter = mock
        controller = main.AppController.__new__(main.AppController)
        controller.vehicle_control = service
        controller.real_mission = {
            "metadata": {"origin_geo": {"lat": 56.0, "lon": 92.0}}
        }

        controller.handle_real_pose_payload({
            "lat": 55.7558,
            "lon": 37.6173,
            "valid": True,
            "source": "nmea0183_gnss",
        })

        self.assertFalse(service.get_telemetry().pose_valid)

    def test_can_telemetry_does_not_keep_stale_gps_pose_fresh(self):
        mock = MockVehicleAdapter()
        service = VehicleControlService(
            VehicleAdapterFactory(real_adapter=None, mock_adapter=mock),
            ControlArbiter(),
        )
        service.adapter = mock
        service.pose_stale_ms = 5.0
        service.navigator.stale_pose_ms = 5.0
        service.submit_pose(Pose2D(1.0, 2.0, 3.0, True, "nmea0183_gnss"))
        time.sleep(0.01)

        telemetry = service.get_telemetry()
        telemetry.last_rx_monotonic = time.monotonic()

        self.assertLess(telemetry.age_ms(), 5.0)
        self.assertGreater(telemetry.pose_age_ms(), 5.0)
        self.assertFalse(service._build_readiness().pose_ok)


if __name__ == "__main__":
    unittest.main()
