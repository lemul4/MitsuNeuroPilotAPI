param map = localPath('D:/CARLA_0.9.15/WindowsNoEditor/CarlaUE4/Content/Carla/Maps/OpenDrive/Town05_Opt.xodr')
param carla_map = 'Town05_Opt'
param render = 0
model scenic.simulators.carla.model

EGO_MODEL = "vehicle.lincoln.mkz_2017"
EGO_SPEED = 20
LEAD_CAR_SPEED = 10

spot = new OrientedPoint on network.lane.centerline
lead_car_spot = new OrientedPoint following roadDirection from spot for 15

ego = new Car following roadDirection from spot for -30,
    with blueprint EGO_MODEL,
    with rolename "hero"

lead_car = new Car following roadDirection from lead_car_spot,
    with blueprint "vehicle.audi.tt",
    with targetSpeed LEAD_CAR_SPEED,
    with rolename "lead"

require (distance to intersection) > 50
terminate when (distance to spot) > 100
