param map = localPath('D:/CARLA_0.9.15/WindowsNoEditor/CarlaUE4/Content/Carla/Maps/OpenDrive/Town05_Opt.xodr')
param carla_map = 'Town05_Opt'
param render = 0
model scenic.simulators.carla.model

EGO_MODEL = "vehicle.lincoln.mkz_2017"
EGO_SPEED = 10
OBSTACLE_DISTANCE = Range(10, 20)

spot = new OrientedPoint on network.lane.centerline
obstacle_spot = new OrientedPoint following roadDirection from spot for OBSTACLE_DISTANCE

ego = new Car following roadDirection from spot for -30,
    with blueprint EGO_MODEL,
    with rolename "hero"

obstacle = new StaticObject at obstacle_spot, 
    with blueprint "static.prop.trafficcone"

require (distance to intersection) > 50
terminate when (distance to spot) > 100
