param map = localPath('C:/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive/Town05_Opt.xodr')
param carla_map = 'Town05_Opt'
param render = 0
model scenic.simulators.carla.model

EGO_MODEL = "vehicle.lincoln.mkz_2017"
EGO_SPEED = 25
SLOW_CAR_SPEED = 10

lane = Uniform(*network.lanes)
spot = new OrientedPoint on lane.centerline
slow_car_spot = new OrientedPoint following roadDirection from spot for 20

ego = new Car following roadDirection from spot for -40,
    with blueprint EGO_MODEL,
    with rolename "hero"

slow_car = new Car following roadDirection from slow_car_spot,
    with blueprint "vehicle.tesla.model3",
    with targetSpeed SLOW_CAR_SPEED,
    with rolename "slow",


require (distance to intersection) > 50
terminate when (distance to spot) > 100
    