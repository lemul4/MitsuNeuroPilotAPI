# hardware/serial_comm.py
import asyncio
import serial_asyncio
from PySide6.QtCore import QObject, Signal
import utils

BUFFER_SIZE = 1024

class SerialManager(QObject):
    data_received = Signal(object) # Передает декодированный пакет (Serial_Data)
    connection_status = Signal(bool, str) # Статус подключения (True/False, Сообщение)
    
    def __init__(self):
        super().__init__()
        self.transport = None
        self.protocol = None
        self.running = False
        self.buffer = utils.CircularBuffer(BUFFER_SIZE)
        self.cmd_queue = asyncio.Queue()

    async def connect_serial(self, port, baudrate=1000000):
        if self.running:
            self.close()
            
        try:
            # Создаем асинхронное соединение
            self.transport, self.protocol = await serial_asyncio.create_serial_connection(
                asyncio.get_event_loop(),
                lambda: SerialProtocol(self),
                url=port,
                baudrate=baudrate
            )
            self.running = True
            self.connection_status.emit(True, f"Подключено к {port}")
            
            # Запускаем цикл обработки исходящих команд
            asyncio.create_task(self._process_commands_loop())
        except Exception as e:
            self.connection_status.emit(False, f"Ошибка: {str(e)}")

    def close(self):
        self.running = False
        if self.transport:
            self.transport.close()
            self.transport = None

    def send_command(self, cmd_obj):
        """Потокобезопасная постановка команды в очередь на отправку"""
        if self.running:
            asyncio.create_task(self.cmd_queue.put(cmd_obj))

    async def _process_commands_loop(self):
        while self.running:
            try:
                cmd = await self.cmd_queue.get()
                if self.transport:
                    self.transport.write(cmd.bytes())
                self.cmd_queue.task_done()
            except Exception as e:
                print(f"Serial Write Error: {e}")

    def _handle_raw_bytes(self, data):
        """Логика разбора байтов (перенесена из исходного main.py)"""
        for b in data:
            self.buffer.add(b)
            if self.buffer.count >= utils.PACKET_SIZE:
                ret = self.buffer.check_buffer()
                if ret:
                    try:
                        pkt = utils.Serial_Data(ret)
                        self.data_received.emit(pkt)
                        self.buffer.remove(utils.PACKET_SIZE)
                    except Exception:
                        pass
                else:
                    if self.buffer.count == self.buffer.size:
                        self.buffer.remove(1)

class SerialProtocol(asyncio.Protocol):
    def __init__(self, manager):
        self.manager = manager

    def data_received(self, data):
        self.manager._handle_raw_bytes(data)

    def connection_lost(self, exc):
        self.manager.connection_status.emit(False, "Соединение потеряно")