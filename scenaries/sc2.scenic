""" Scenario Description
уклонение от статичного препядсвия

To run this file using the Carla simulator:
    scenic examples/carla/manual_control/carlaChallenge5_dynamic.scenic --2d --model scenic.simulators.carla.model --simulate
"""
param map = localPath('C:/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive/Town05_Opt.xodr')
param carla_map = 'Town05_Opt'
param render = 0
model scenic.simulators.carla.model

EGO_MODEL = "vehicle.lincoln.mkz_2017"
EGO_SPEED = 10
OBSTACLE_DISTANCE = Range(5, 15)

lane = Uniform(*network.lanes)
spot = new OrientedPoint on lane.centerline
obstacle_spot = new OrientedPoint following roadDirection from spot for OBSTACLE_DISTANCE

ego = new Car following roadDirection from spot for -30,
    with blueprint EGO_MODEL,
    with rolename "hero"

obstacle = new Prop at obstacle_spot,
    with blueprint "static.prop.streetbarrier"


require (distance to intersection) > 30
terminate when (distance to spot) > 50
