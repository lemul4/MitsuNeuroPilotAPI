""" Scenario Description
убрать дублирует сценарий 1

To run this file using the Carla simulator:
    scenic examples/carla/manual_control/carlaChallenge5_dynamic.scenic --2d --model scenic.simulators.carla.model --simulate
"""
param map = localPath('C:/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive/Town05_Opt.xodr')
param carla_map = 'Town05_Opt'
param render = 0
model scenic.simulators.carla.model

EGO_MODEL = "vehicle.lincoln.mkz_2017"
EGO_SPEED = 15
PEDESTRIAN_SPEED = Range(1.0, 1.5)

lane = Uniform(*network.lanes)
spot = new OrientedPoint on lane.centerline

pedestrian_spot = new OrientedPoint following roadDirection from spot for 10

ego = new Car following roadDirection from spot for -30,
    with blueprint EGO_MODEL,
    with rolename "hero"

pedestrian = new Pedestrian left of pedestrian_spot by 3,
    with heading 90 deg relative to pedestrian_spot.heading,
    with behavior CrossingBehavior(ego, PEDESTRIAN_SPEED, 10)

terminate when (distance to spot) > 100
