#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an NPC agent to control the ego vehicle
"""

from __future__ import print_function

import carla
# Импортируем BehaviorAgent
from agents.navigation.behavior_agent import BehaviorAgent
# Импорт BasicAgent больше не нужен
# from agents.navigation.basic_agent import BasicAgent
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
    _hero_actor = None # Добавим ссылку на hero actor

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self.track = Track.SENSORS

        self._agent = None
        self._hero_actor = None # Инициализируем здесь тоже

    def sensors(self):
        # Ваши сенсоры (оставьте те, что вам нужны для leaderboard)
        sensors = [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},
            # Добавьте другие сенсоры, если они требуются вашим заданием leaderboard
        ]
        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        # Находим hero actor, если еще не нашли
        if not self._hero_actor:
             for actor in CarlaDataProvider.get_world().get_actors():
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                    self._hero_actor = actor
                    break

        # Если hero actor не найден, не можем продолжить
        if not self._hero_actor:
             print("NpcAgent: Hero actor not found!") # Опционально: добавить лог
             return carla.VehicleControl()

        # Инициализируем агента и устанавливаем план, если еще не сделано
        if not self._agent:
            # Инициализируем BehaviorAgent
            # BehaviorAgent унаследует метод trace_route от BasicAgent
            self._agent = BehaviorAgent(self._hero_actor) # Можете добавить behavior='cautious'/'aggressive'

            # --- ВОССТАНОВЛЕНА ЛОГИКА ПОСТРОЕНИЯ ПЛОТНОГО ПЛАНА ЧЕРЕЗ trace_route ---
            # Получаем глобальный план из leaderboard (список пар (transform, road_option))
            global_plan = self._global_plan_world_coord # Это исходный разреженный план от leaderboard

            # План для агента (будет плотным)
            plan_with_waypoints = []
            prev_wp = None
            world_map = CarlaDataProvider.get_map() # Получаем текущую карту

            # Проходим по всем ключевым точкам глобального плана от leaderboard
            for transform, road_option in global_plan:
                # Получаем Waypoint, соответствующий текущей Transform
                current_wp = world_map.get_waypoint(transform.location)

                # Если у нас есть предыдущая точка, строим плотный маршрут между prev_wp и current_wp
                # trace_route генерирует список промежуточных вейпоинтов на дорожной сети
                if prev_wp:
                    # trace_route возвращает список пар (waypoint, road_option)
                    # Используем _agent.trace_route, так как BehaviorAgent унаследовал этот метод
                    segment_plan = self._agent.trace_route(prev_wp, current_wp)
                    plan_with_waypoints.extend(segment_plan)
                else:
                    # Для самой первой точки просто добавляем ее в план (это начало маршрута)
                    plan_with_waypoints.append((current_wp, road_option))


                prev_wp = current_wp # Обновляем предыдущую точку для следующей итерации

            # Устанавливаем построенный ПЛОТНЫЙ план для BehaviorAgent
            # BehaviorAgent будет использовать этот плотный список вейпоинтов для навигации
            self._agent.set_global_plan(plan_with_waypoints)
            # --- КОНЕЦ ЛОГИКИ ПОСТРОЕНИЯ ПЛОТНОГО ПЛАНА ---


            # Возвращаем пустое управление на первом шаге
            return carla.VehicleControl()

        # Если агент инициализирован, выполняем один шаг навигации с помощью BehaviorAgent
        else:
            # BehaviorAgent.run_step() использует свой локальный планировщик и логику поведения
            # для следования по установленному ПЛОТНОМУ плану.
            return self._agent.run_step()