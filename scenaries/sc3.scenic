param map = localPath('D:/CARLA_0.9.15/WindowsNoEditor/CarlaUE4/Content/Carla/Maps/OpenDrive/Town05_Opt.xodr')
param carla_map = 'Town05_Opt'
param render = 0
model scenic.simulators.carla.model

EGO_MODEL = "vehicle.lincoln.mkz_2017"
EGO_SPEED = 15
PEDESTRIAN_SPEED = Range(1.0, 1.5)

spot = new OrientedPoint on network.lane.centerline
pedestrian_spot = new OrientedPoint following roadDirection from spot for 10

ego = new Car following roadDirection from spot for -30,
    with blueprint EGO_MODEL,
    with rolename "hero"

pedestrian = new Pedestrian left of pedestrian_spot by 3,
    with heading 90 deg relative to pedestrian_spot.heading,
    with behavior CrossingBehavior(ego, PEDESTRIAN_SPEED, 10)

require (distance to intersection) > 50
terminate when (distance to spot) > 100
