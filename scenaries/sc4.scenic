param map = localPath('C:/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive/Town05_Opt.xodr')
param carla_map = 'Town05_Opt'
param render = 0
model scenic.simulators.carla.model

EGO_MODEL = "vehicle.lincoln.mkz_2017"
EGO_SPEED = 20
LEAD_CAR_SPEED = 60

lane = Uniform(*network.lanes)
spot = new OrientedPoint on lane.centerline
lead_car_spot = new OrientedPoint following roadDirection from spot for 15

ego = new Car following roadDirection from spot for -30,
    with blueprint EGO_MODEL,
    with rolename "hero"

lead_car = new Car at lead_car_spot,
    with blueprint "vehicle.audi.tt",
    with targetSpeed LEAD_CAR_SPEED,
    with behavior BuiltinVehicleBehavior








require (distance to intersection) > 25
terminate when (distance to spot) > 100
