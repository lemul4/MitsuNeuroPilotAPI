#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an NPC agent to control the ego vehicle
"""

from __future__ import print_function

import carla
from agents.navigation.behavior_agent import BehaviorAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track


def get_entry_point():
    return 'NpcAgent'


class NpcAgent(AutonomousAgent):

    """
    NPC autonomous agent to control the ego vehicle
    """

    _agent = None
    _route_assigned = False

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        # No additional config needed here
        pass

    def sensors(self):
        """
        Define the sensor suite required by the agent
        """
        return [
            {
                'type': 'sensor.camera.rgb',
                'x': 0.7, 'y': -0.4, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 300, 'height': 200, 'fov': 100,
                'id': 'Left'
            },
        ]

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        # Initialize BasicAgent and assign route once
        if not self._agent:
            # Find the ego (hero) actor
            hero_actor = None
            for actor in CarlaDataProvider.get_world().get_actors():
                if actor.attributes.get('role_name', '') == 'hero':
                    hero_actor = actor
                    break

            if not hero_actor:
                return carla.VehicleControl()

            # Create BasicAgent with desired speed
            self._agent = BehaviorAgent(hero_actor, "cautious")

            # Use parent to downsample and store global plan
            super().set_global_plan(self._global_plan, self._global_plan_world_coord)

            # Convert stored world-coordinate route transforms into Waypoints
            route = []
            map_api = CarlaDataProvider.get_map()
            for transform, road_option in self._global_plan_world_coord:
                wp = map_api.get_waypoint(transform.location)
                route.append((wp, road_option))

            # Assign the processed route to BasicAgent
            self._agent.set_global_plan(route)

            self._route_assigned = True
            return carla.VehicleControl()

        # Agent is ready: run control step
        return self._agent.run_step()