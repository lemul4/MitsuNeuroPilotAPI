# hardware/can_commands.py
import time
import utils

class CANCommandFactory:
    """Генерирует объекты Serial_Data для отправки в машину"""
    
    @staticmethod
    def create_base_packet(hex_str):
        return utils.Serial_Data(bytearray.fromhex(hex_str))

    @staticmethod
    def prepare_packet(cmd_obj, value, data_idx=1):
        """Заполняет пакет актуальными данными и временем"""
        cmd_obj.TIME = int(time.time())
        # Инкремент счетчика пакетов (CNC) для контроля доставки
        cmd_obj.CAN_DATA.CNC = (cmd_obj.CAN_DATA.CNC + 1) & 0xFF
        cmd_obj.CAN_DATA.DATA[0] = 0x01 # Флаг активной команды
        cmd_obj.CAN_DATA.DATA[data_idx] = value
        cmd_obj.store_crc8()
        return cmd_obj

# Преднастроенные шаблоны
GEAR_TEMPLATE = "AA 00000000 3300 00 02 01 00 00 00 01 00 00"
ACCEL_TEMPLATE = "AA 00000000 3800 00 02 01 00 00 00 01 00 00"
BRAKE_TEMPLATE = "AA 00000000 3700 00 02 01 00 00 00 01 00 00"
ANGLE_TEMPLATE = "AA 00000000 3200 00 02 01 00 00 00 01 00 00"
CRUISE_TEMPLATE = "AA 00000000 7700 00 02 01 00 00 00 01 00 00"