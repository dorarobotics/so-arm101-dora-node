"""Feetech SCS/STS serial-bus protocol for the STS3215 servos in an SO-ARM101.

One ``FeetechBus`` owns one serial line. Every servo on that line (both arms of a
Nano and its three wheel servos) must go through the same bus object, because two
processes writing to one half-duplex line corrupt each other's packets.

Packet layout (instruction):  FF FF id len instr params... checksum
Packet layout (status):       FF FF id len error params... checksum
``len`` counts instr/error + params + checksum; ``checksum = ~(id + len + body) & 0xFF``.

Register addresses and encodings follow Hugging Face LeRobot v0.4.3 (Apache-2.0),
as already used by DoRobot Studio's native SO-101 capture driver.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from typing import Protocol

BROADCAST_ID = 0xFE

INSTR_PING = 0x01
INSTR_READ = 0x02
INSTR_WRITE = 0x03
INSTR_SYNC_WRITE = 0x83

STS3215_MODEL = 777
RESOLUTION = 4096  # 12-bit magnetic encoder

# Control table (address, size in bytes).
MODEL_NUMBER = (3, 2)
RETURN_DELAY_TIME = (7, 1)
MIN_POSITION_LIMIT = (9, 2)
MAX_POSITION_LIMIT = (11, 2)
MAX_TORQUE_LIMIT = (16, 2)
P_COEFFICIENT = (21, 1)
D_COEFFICIENT = (22, 1)
I_COEFFICIENT = (23, 1)
PROTECTION_CURRENT = (28, 2)
HOMING_OFFSET = (31, 2)
OPERATING_MODE = (33, 1)
OVERLOAD_TORQUE = (36, 1)
TORQUE_ENABLE = (40, 1)
ACCELERATION = (41, 1)
GOAL_POSITION = (42, 2)
GOAL_SPEED = (46, 2)  # wheel mode: sign-magnitude, bit 15 = reverse
LOCK = (55, 1)
PRESENT_POSITION = (56, 2)
PRESENT_LOAD = (60, 2)
PRESENT_VOLTAGE = (62, 1)
PRESENT_TEMPERATURE = (63, 1)
MOVING = (66, 1)
PRESENT_CURRENT = (69, 2)

# Encodings and units from the Feetech STS memory table, SRAM feedback section
# (https://ftservo.github.io/ftservo-wiki/en/reference/parameter/memory-sts/#24-sram):
# - Present_Load (60): "duty cycle of the output driving the motor; BIT10 is the direction
#   bit", 0.1 % per unit. It is the PWM duty of the motor drive, an effort proxy, not a
#   torque measurement. LeRobot decodes it with the same sign bit 10.
# - Present_Current (69): "motor phase current feedback; BIT15 is the current direction
#   bit", 6.5 mA per unit. Signed, like position and speed.
LOAD_SIGN_BIT = 10
LOAD_PERCENT_PER_UNIT = 0.1
CURRENT_SIGN_BIT = 15
CURRENT_MA_PER_UNIT = 6.5

STATUS_ERROR_BITS = {
    0x01: "input voltage",
    0x02: "angle limit",
    0x04: "overheat",
    0x08: "range",
    0x10: "checksum",
    0x20: "overload",
    0x40: "instruction",
}


# What the serial transport itself raises when the USB-serial link fails. pyserial's
# SerialException is an OSError, but its posix backend also lets termios.error escape
# (e.g. reset_input_buffer -> tcflush after the adapter disappears), and termios.error is
# not an OSError. Both are raised as FeetechError ("transport error") by the bus.
try:
    import termios

    TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (OSError, termios.error)
except ImportError:  # no termios on this platform
    TRANSPORT_ERRORS = (OSError,)


class FeetechError(RuntimeError):
    """A servo did not answer, answered malformed bytes, or reported a fault."""


class Transport(Protocol):
    """The subset of ``serial.Serial`` the bus needs. Also implemented by the simulator."""

    def write(self, data: bytes) -> int | None: ...

    def read(self, size: int) -> bytes: ...

    def reset_input_buffer(self) -> None: ...


def checksum(body: Iterable[int]) -> int:
    return (~sum(body)) & 0xFF


def build_packet(servo_id: int, instruction: int, params: bytes = b"") -> bytes:
    length = len(params) + 2
    body = bytes([servo_id, length, instruction]) + params
    return b"\xff\xff" + body + bytes([checksum(body)])


def encode_sign_magnitude(value: int, sign_bit: int) -> int:
    magnitude = abs(value)
    if magnitude >= 1 << sign_bit:
        raise ValueError(f"{value} does not fit in {sign_bit} magnitude bits")
    return magnitude | (1 << sign_bit) if value < 0 else magnitude


def decode_sign_magnitude(raw: int, sign_bit: int) -> int:
    magnitude = raw & ((1 << sign_bit) - 1)
    return -magnitude if raw & (1 << sign_bit) else magnitude


def _to_bytes(value: int, size: int) -> bytes:
    if not 0 <= value < 1 << (8 * size):
        raise ValueError(f"{value} does not fit in {size} byte(s)")
    return value.to_bytes(size, "little")


class FeetechBus:
    """Half-duplex Feetech bus with acknowledged reads and writes.

    Every unicast exchange checks the status packet's header, id, length, checksum
    and error byte, so a wrong or silent servo raises ``FeetechError`` instead of
    being mistaken for success. Errors from the serial transport itself (a USB
    adapter unplugged, an I/O error: pyserial's ``SerialException``, an ``OSError``,
    or a raw ``termios.error`` from its posix backend; see ``TRANSPORT_ERRORS``) are
    raised as ``FeetechError`` too, so every caller's bus-error handling applies and
    nothing escapes the node's event loop.
    """

    def __init__(self, transport: Transport, *, retries: int = 1) -> None:
        self._io = transport
        self._retries = retries

    @classmethod
    def open_serial(cls, port: str, baudrate: int = 1_000_000, timeout_s: float = 0.02) -> FeetechBus:
        import serial  # lazy: tests and the simulator do not need pyserial

        return cls(serial.Serial(port, baudrate=baudrate, timeout=timeout_s))

    def close(self) -> None:
        close = getattr(self._io, "close", None)
        if callable(close):
            close()

    # -- low level ---------------------------------------------------------

    def _exchange(self, servo_id: int, instruction: int, params: bytes, reply_len: int) -> bytes:
        last: Exception | None = None
        for _ in range(self._retries + 1):
            try:
                return self._exchange_once(servo_id, instruction, params, reply_len)
            except FeetechError as e:
                if "reported" in str(e):  # a real fault, not a transmission glitch
                    raise
                last = e
                time.sleep(0.002)
        assert last is not None
        raise last  # the last attempt's error, with its cause (e.g. the SerialException) kept

    def _exchange_once(self, servo_id: int, instruction: int, params: bytes, reply_len: int) -> bytes:
        expected = reply_len + 6
        try:
            self._io.reset_input_buffer()
            self._io.write(build_packet(servo_id, instruction, params))
            status = self._io.read(expected)
        except TRANSPORT_ERRORS as e:
            raise FeetechError(f"servo {servo_id}: transport error: {e}") from e
        if len(status) != expected:
            raise FeetechError(f"servo {servo_id}: no reply ({len(status)}/{expected} bytes)")
        if status[0:2] != b"\xff\xff" or status[2] != servo_id or status[3] != reply_len + 2:
            raise FeetechError(f"servo {servo_id}: malformed reply {status.hex()}")
        if checksum(status[2:-1]) != status[-1]:
            raise FeetechError(f"servo {servo_id}: bad reply checksum")
        if status[4]:
            faults = [name for bit, name in STATUS_ERROR_BITS.items() if status[4] & bit]
            raise FeetechError(f"servo {servo_id} reported {', '.join(faults) or hex(status[4])}")
        return bytes(status[5:-1])

    # -- public API --------------------------------------------------------

    def ping(self, servo_id: int) -> bool:
        try:
            self._exchange(servo_id, INSTR_PING, b"", 0)
        except FeetechError:
            return False
        return True

    def read(self, servo_id: int, register: tuple[int, int]) -> int:
        address, size = register
        data = self._exchange(servo_id, INSTR_READ, bytes([address, size]), size)
        return int.from_bytes(data, "little")

    def write(self, servo_id: int, register: tuple[int, int], value: int) -> None:
        address, size = register
        self._exchange(servo_id, INSTR_WRITE, bytes([address]) + _to_bytes(value, size), 0)

    def sync_write(self, register: tuple[int, int], values: Mapping[int, int]) -> None:
        """Write one register on several servos in a single broadcast packet (no reply)."""
        if not values:
            return
        address, size = register
        params = bytearray([address, size])
        for servo_id, value in values.items():
            params.append(servo_id)
            params += _to_bytes(value, size)
        try:
            self._io.write(build_packet(BROADCAST_ID, INSTR_SYNC_WRITE, bytes(params)))
        except TRANSPORT_ERRORS as e:
            raise FeetechError(f"sync write to {sorted(values)}: transport error: {e}") from e

    def read_position(self, servo_id: int) -> int:
        return decode_sign_magnitude(self.read(servo_id, PRESENT_POSITION), 15)

    def read_load(self, servo_id: int) -> int:
        """Signed motor-drive duty cycle in raw units (0.1 % each); the sign is the direction."""
        return decode_sign_magnitude(self.read(servo_id, PRESENT_LOAD), LOAD_SIGN_BIT)

    def read_current(self, servo_id: int) -> int:
        """Signed motor phase current in raw units (6.5 mA each)."""
        return decode_sign_magnitude(self.read(servo_id, PRESENT_CURRENT), CURRENT_SIGN_BIT)

    def read_homing_offset(self, servo_id: int) -> int:
        return decode_sign_magnitude(self.read(servo_id, HOMING_OFFSET), 11)

    def write_homing_offset(self, servo_id: int, offset: int) -> None:
        self.write(servo_id, HOMING_OFFSET, encode_sign_magnitude(offset, 11))
