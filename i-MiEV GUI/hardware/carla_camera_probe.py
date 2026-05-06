"""
Safe CARLA camera probe for MitsuNeuroPilot.
Run while CARLA/scenario is running:
    python carla_camera_probe.py --host 127.0.0.1 --port 2000 --role hero
It does not call sensor.listen() and does not modify the world.
"""
import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--role", action="append", default=None, help="Ego role_name. Can be used multiple times.")
    args = parser.parse_args()

    role_names = tuple(args.role or ["hero", "ego_vehicle", "ego", "player"])

    try:
        import carla
    except Exception as exc:
        print(f"CARLA PROBE ERROR: cannot import carla: {exc}")
        return 2

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        world = client.get_world()
    except Exception as exc:
        print(f"CARLA PROBE ERROR: cannot connect to {args.host}:{args.port}: {exc}")
        return 2

    actors = world.get_actors()
    vehicles = list(actors.filter("vehicle.*"))
    cameras = list(actors.filter("sensor.camera.*"))

    ego = None
    for vehicle in vehicles:
        role = vehicle.attributes.get("role_name", "")
        if role in role_names:
            ego = vehicle
            break

    print(f"CARLA PROBE: map={world.get_map().name}")
    print(f"CARLA PROBE: vehicles={len(vehicles)} cameras={len(cameras)} role_filter={role_names}")

    if ego is None:
        print("CARLA PROBE: ego vehicle not found")
        for vehicle in vehicles[:25]:
            print(f"  vehicle id={vehicle.id} type={vehicle.type_id} role={vehicle.attributes.get('role_name', '-')}")
        return 1

    print(
        "CARLA PROBE: ego found "
        f"id={ego.id} type={ego.type_id} role={ego.attributes.get('role_name', '-')}"
    )

    attached = []
    for camera in cameras:
        parent = getattr(camera, "parent", None)
        if parent is not None and parent.id == ego.id:
            attached.append(camera)

    if not attached:
        print("CARLA PROBE: no camera sensors attached to ego")
        if cameras:
            print("CARLA PROBE: other camera sensors:")
            for camera in cameras[:25]:
                parent = getattr(camera, "parent", None)
                parent_id = parent.id if parent is not None else None
                print(
                    f"  camera id={camera.id} type={camera.type_id} "
                    f"role={camera.attributes.get('role_name', '-')} parent={parent_id} "
                    f"listening={camera.is_listening}"
                )
        return 1

    print(f"CARLA PROBE: ego cameras={len(attached)}")
    for camera in attached:
        tf = camera.get_transform()
        loc = tf.location
        rot = tf.rotation
        print(
            f"  camera id={camera.id} type={camera.type_id} "
            f"role={camera.attributes.get('role_name', '-')} "
            f"listening={camera.is_listening} "
            f"image={camera.attributes.get('image_size_x', '?')}x{camera.attributes.get('image_size_y', '?')} "
            f"fov={camera.attributes.get('fov', '?')} "
            f"loc=({loc.x:.2f},{loc.y:.2f},{loc.z:.2f}) "
            f"rot=({rot.pitch:.1f},{rot.yaw:.1f},{rot.roll:.1f})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
