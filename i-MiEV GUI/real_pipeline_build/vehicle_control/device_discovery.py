from __future__ import annotations

from typing import List

from .models import DeviceDescriptor, DeviceKind


def discover_vehicle_devices(include_test_devices: bool = True) -> List[DeviceDescriptor]:
    devices: List[DeviceDescriptor] = []
    devices.append(DeviceDescriptor("virtual-demo", "VIRTUAL_DEMO_MODE", DeviceKind.VIRTUAL_DEMO))
    if include_test_devices:
        devices.append(DeviceDescriptor("test-mock-vehicle", "TEST_MOCK_VEHICLE", DeviceKind.MOCK_VEHICLE))
        devices.append(DeviceDescriptor("test-replay-log", "TEST_REPLAY_LOG", DeviceKind.REPLAY_LOG))
        devices.append(DeviceDescriptor("test-serial-loopback", "TEST_SERIAL_LOOPBACK", DeviceKind.SERIAL_LOOPBACK))

    try:
        import serial.tools.list_ports
        for port in serial.tools.list_ports.comports():
            label = port.device
            description = getattr(port, "description", "") or ""
            if description and description not in label:
                label = f"{description} - {port.device}"
            devices.append(DeviceDescriptor(
                id=f"serial:{port.device}",
                label=label,
                kind=DeviceKind.REAL_SERIAL,
                port=port.device,
                metadata={"description": description},
            ))
    except Exception:
        pass
    return devices
