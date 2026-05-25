# hardware/serial_comm.py
import asyncio
import concurrent.futures
import serial_asyncio
from PySide6.QtCore import QObject, Signal
import utils

BUFFER_SIZE = 1024


class SerialManager(QObject):
    data_received = Signal(object)          # decoded Serial_Data packet
    connection_status = Signal(bool, str)   # connected flag, message

    def __init__(self):
        super().__init__()
        self.transport = None
        self.protocol = None
        self.running = False
        self.buffer = utils.CircularBuffer(BUFFER_SIZE)
        self.cmd_queue = asyncio.Queue()

        # Real-time control path. Service/diagnostic commands still use
        # cmd_queue; steering/throttle/brake should use these latest-frame
        # helpers or hardware.command_scheduler.ControlCommandScheduler.
        self._latest_control_packet_set = None
        self._control_flush_scheduled = False
        self._writer_future = None


    def _schedule_coro_threadsafe(self, coro, label="serial task"):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError as exc:
                try:
                    coro.close()
                except Exception:
                    pass
                print(f"Serial Async Error: cannot schedule {label}: {exc}")
                return None
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception as exc:
            try:
                coro.close()
            except Exception:
                pass
            print(f"Serial Async Error: failed to schedule {label}: {exc}")
            return None

        def _done(done_future):
            try:
                done_future.result()
            except (asyncio.CancelledError, concurrent.futures.CancelledError):
                pass
            except Exception as exc:
                if self.running:
                    print(f"Serial Async Error in {label}: {exc}")
        future.add_done_callback(_done)
        return future

    async def connect_serial(self, port, baudrate=1000000):
        if self.running:
            self.close()

        try:
            self.transport, self.protocol = await serial_asyncio.create_serial_connection(
                asyncio.get_event_loop(),
                lambda: SerialProtocol(self),
                url=port,
                baudrate=baudrate,
            )
            self.running = True
            self.connection_status.emit(True, f"Подключено к {port}")
            self._writer_future = self._schedule_coro_threadsafe(self._process_commands_loop(), "serial_writer")
        except Exception as e:
            self.connection_status.emit(False, f"Ошибка: {str(e)}")

    def close(self):
        self.running = False
        self._latest_control_packet_set = None
        self._control_flush_scheduled = False
        writer_future = self._writer_future
        self._writer_future = None
        if writer_future is not None and not writer_future.done():
            try:
                writer_future.cancel()
            except Exception:
                pass
        if self.transport:
            self.transport.close()
            self.transport = None

    def send_command(self, cmd_obj):
        """Queue a service/diagnostic command. Keeps legacy behavior."""
        if not self.running:
            return
        try:
            self.cmd_queue.put_nowait(cmd_obj)
        except Exception as exc:
            # Fallback: do not lose safety/service packets if the asyncio queue
            # is temporarily unavailable. Serial transports are used from the Qt
            # main thread in this application.
            if self.transport:
                try:
                    self.transport.write(cmd_obj.bytes())
                except Exception as write_exc:
                    print(f"Serial Queue Error: {exc}; direct write failed: {write_exc}")

    def write_packet_immediate(self, cmd_obj):
        """Write a control packet without the FIFO queue.

        This is intended for the fixed-rate vehicle-control scheduler. It does
        not store old commands, so stale throttle/steering frames cannot build
        up behind slower service commands.
        """
        if self.running and self.transport:
            self.transport.write(cmd_obj.bytes())

    def send_control_command_latest(self, cmd_obj):
        self.send_control_packet_set_latest([cmd_obj])

    def send_control_packet_set_latest(self, cmd_objs):
        """Coalesce bursty control output: latest packet set wins.

        This is deliberately synchronous: control packets must not wait behind a
        Qt/async scheduling timer, and old control packets must not accumulate.
        """
        if not self.running or not self.transport:
            return
        packets = list(cmd_objs or [])
        self._latest_control_packet_set = packets
        try:
            for cmd in packets:
                self.transport.write(cmd.bytes())
        except Exception as exc:
            print(f"Serial Control Write Error: {exc}")
        finally:
            self._latest_control_packet_set = None
            self._control_flush_scheduled = False

    async def _flush_latest_control_packet_set(self):
        try:
            await asyncio.sleep(0)
            packets = self._latest_control_packet_set or []
            self._latest_control_packet_set = None
            if self.running and self.transport:
                for cmd in packets:
                    self.transport.write(cmd.bytes())
        except Exception as e:
            print(f"Serial Control Write Error: {e}")
        finally:
            self._control_flush_scheduled = False
            # If a newer packet set arrived while flushing, schedule one more
            # flush. This keeps only the newest set, not a FIFO backlog.
            if self.running and self._latest_control_packet_set is not None:
                self._control_flush_scheduled = True
                self._schedule_coro_threadsafe(self._flush_latest_control_packet_set(), "serial_control_flush")

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
        self.manager.running = False
        self.manager.connection_status.emit(False, "Соединение потеряно")
