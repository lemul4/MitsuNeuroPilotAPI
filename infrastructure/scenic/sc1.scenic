"""
Сценарий обгона медленной машины
"""

param map = localPath('C:/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr')
param carla_map = 'Town03'
param render = 0
param timestep = 0.025
model scenic.simulators.carla.model


MODEL_EGO = 'vehicle.audi.a2'
MODEL_adversary = 'vehicle.lincoln.mkz_2017'
MAX_DISTANCE = 50
param ADV_DIST = VerifaiRange(10, 15)
param ADV_SPEED = VerifaiRange(2, 5)

INIT_DIST = 50


initLane = Uniform(*network.lanes)
egoSpawnPt = new OrientedPoint in initLane.centerline


ego = new Car at egoSpawnPt,
    with blueprint MODEL_EGO,
    with rolename "hero"

adversary = new Car following roadDirection for globalParameters.ADV_DIST,
    with blueprint MODEL_adversary,
    with behavior FollowLaneBehavior(target_speed=globalParameters.ADV_SPEED)

require (distance to intersection) > INIT_DIST
require (distance from adversary to intersection) > INIT_DIST
require always (adversary.laneSection._fasterLane is not None)

terminate when (distance from ego to adversary) > MAX_DISTANCE
terminate when simulation().currentTime >= 500

