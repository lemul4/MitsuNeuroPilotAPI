#!/usr/bin/python3
# utils.py
import xml.etree.ElementTree as ET
import os

PACKET_SIZE = 16

def scan_carla_routes(project_root):
    routes_list = []
    # Убедимся, что пути строятся относительно корня MitsuNeuroPilotAPI
    folders = {
        "TEST": os.path.join(project_root, "data", "benchmark_routes"),
        "TRAIN": os.path.join(project_root, "data", "data_routes")
    }

    print(f"[DEBUG] Scanning routes in: {project_root}")

    for tag, base_path in folders.items():
        if not os.path.exists(base_path):
            print(f"[DEBUG] Path not found: {base_path}")
            continue
            
        for root_dir, _, files in os.walk(base_path):
            for file in files:
                if file.endswith(".xml"):
                    full_path = os.path.join(root_dir, file)
                    try:
                        tree = ET.parse(full_path)
                        root = tree.getroot()
                        
                        # В некоторых CARLA XML структура может отличаться
                        # Пробуем найти город (town) более гибко
                        town = "Unknown"
                        route_node = root.find('.//route') # Поиск по всему дереву
                        if route_node is not None:
                            town = route_node.get('town', "Unknown")
                        
                        routes_list.append({
                            "label": f"[{tag}] {file} ({town})",
                            "path": full_path
                        })
                        print(f"[DEBUG] Found route: {file}")
                    except Exception as e:
                        print(f"[DEBUG] XML Error in {file}: {e}")
    
    if not routes_list:
        print("[DEBUG] No .xml files found in target folders!")
    return routes_list

def calc_crc8(data: bytearray):
    crc8 = 0xFF
    for byte in data:
        crc8 ^= byte
        for i in range(8):
            xor_val = 0x07 if (crc8 & 0x80) else 0x00
            crc8 = ((crc8 << 1) & 0x00FF) ^ xor_val
    return crc8

class CAN_Data:
    def __init__(self, data):
        if len(data) < 8:
            # Заполняем нулями, если данных мало, чтобы не падало
            data = list(data) + [0]*(8-len(data))
        self.CNC = data[0]
        self.TYPE = data[1]
        self.DATA = []
        self.DATA.append(data[2])
        self.DATA.append(data[3])
        self.DATA.append(data[4])
        self.DATA.append(data[5])
        self.STATE = data[6]
        self.CRC = data[7]

    def store_crc8(self):
        self.CRC = calc_crc8(self.bytes()[:-1])

    def show(self):
        return [self.CNC, self.TYPE, self.DATA, self.STATE, self.CRC]

    def bytes(self):
        return bytearray([self.CNC, self.TYPE, self.DATA[0], self.DATA[1], self.DATA[2], self.DATA[3], self.STATE, self.CRC])

class Serial_Data:
    def __init__(self, data):
        if len(data) < PACKET_SIZE:
             data = list(data) + [0]*(PACKET_SIZE-len(data))
        self.START_BYTE = 0xAA
        self.TIME = data[1] + (data[2] << 8) + (data[3] << 16) + (data[4] << 24)
        self.CAN_ID = data[5] + (data[6] << 8)
        items = []
        idx = 7
        for _ in range(8):
            items.append(data[idx])
            idx += 1
        self.CAN_DATA = CAN_Data(items)
        self.CRC = data[15]
        
    def store_crc8(self):
        self.CAN_DATA.store_crc8()
        # Пересчитываем CRC всего пакета
        self.CRC = calc_crc8(self.bytes()[:-1])

    def bytes(self):
        header = bytearray([self.START_BYTE, self.TIME&0xff, (self.TIME>>8)&0xff, (self.TIME>>16)&0xff, (self.TIME>>24)&0xff, self.CAN_ID&0xff, (self.CAN_ID>>8)&0xff])
        return header + self.CAN_DATA.bytes() + bytearray([self.CRC])

class CircularBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = [0] * size # Инициализируем нулями
        self.head = 0
        self.tail = 0
        self.count = 0

    def add(self, item):
        self.buffer[self.head] = item
        if self.count == self.size:
            self.tail = (self.tail + 1) % self.size
        else:
            self.count += 1
        self.head = (self.head + 1) % self.size

    def gets_count(self, count):
        if self.count == 0:
            return []
        if count > self.count:
            count = self.count
        items = []
        index = self.tail
        for _ in range(count):
            items.append(self.buffer[index])
            index = (index + 1) % self.size
        return items
    
    def get(self, idx):
        if idx >= self.count:
            return 0
        actual_idx = (self.tail + idx) % self.size
        return self.buffer[actual_idx]

    def remove(self, count):
        for _ in range(min(count, self.count)):
            self.tail = (self.tail + 1) % self.size
            self.count -= 1

    def check_buffer(self):
        if (self.get(0) != 0xAA or self.count < PACKET_SIZE):
            return None
        
        items = self.gets_count(PACKET_SIZE - 1) # берем все кроме CRC
        crc8 = calc_crc8(items)
        
        if (crc8 == self.get(PACKET_SIZE - 1)):
            items.append(crc8)
            return items
        else:
            return None
        

    