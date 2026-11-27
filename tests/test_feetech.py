from __future__ import annotations

import pytest

from so_arm101_node import feetech as ft
from so_arm101_node.sim import SimulatedServos


def test_ping_packet_matches_feetech_reference():
    # FF FF 01 02 01 FB is the documented ping for servo 1.
    assert ft.build_packet(1, ft.INSTR_PING) == bytes.fromhex("ffff010201fb")


def test_sign_magnitude_round_trip():
    for value in (-2047, -1, 0, 1, 2047):
        assert ft.decode_sign_magnitude(ft.encode_sign_magnitude(value, 11), 11) == value
    assert ft.encode_sign_magnitude(-5, 15) == 0x8005
    with pytest.raises(ValueError):
        ft.encode_sign_magnitude(2048, 11)


def test_read_write_through_simulated_servo():
    bus = ft.FeetechBus(SimulatedServos([3]))
    assert bus.read(3, ft.MODEL_NUMBER) == ft.STS3215_MODEL
    bus.write(3, ft.GOAL_POSITION, 1234)
    assert bus.read(3, ft.GOAL_POSITION) == 1234
    bus.write_homing_offset(3, -100)
    assert bus.read_homing_offset(3) == -100
    # present = encoder - offset
    assert bus.read_position(3) == 2048 + 100


def test_missing_servo_raises_and_ping_is_false():
    bus = ft.FeetechBus(SimulatedServos([1]))
    assert bus.ping(1) is True
    assert bus.ping(9) is False
    with pytest.raises(ft.FeetechError, match="no reply"):
        bus.read(9, ft.MODEL_NUMBER)


def test_servo_fault_is_reported_not_retried():
    class Faulty(SimulatedServos):
        def _reply(self, servo_id, payload):
            body = bytes([servo_id, len(payload) + 2, 0x20]) + payload
            self._out += b"\xff\xff" + body + bytes([ft.checksum(body)])

    bus = ft.FeetechBus(Faulty([1]))
    with pytest.raises(ft.FeetechError, match="overload"):
        bus.read(1, ft.MODEL_NUMBER)


def test_corrupt_reply_is_rejected():
    class Corrupt(SimulatedServos):
        def read(self, size):
            data = bytearray(super().read(size))
            if data:
                data[-1] ^= 0xFF
            return bytes(data)

    bus = ft.FeetechBus(Corrupt([1]), retries=0)
    with pytest.raises(ft.FeetechError, match="checksum"):
        bus.read(1, ft.MODEL_NUMBER)


def test_sync_write_reaches_every_listed_servo():
    sim = SimulatedServos([1, 2, 7])
    bus = ft.FeetechBus(sim)
    bus.sync_write(ft.GOAL_POSITION, {1: 100, 7: 300})
    assert sim.goal(1) == 100
    assert sim.goal(2) == 0
    assert sim.goal(7) == 300


def test_load_and_current_decoding():
    """Feetech STS table: Present_Load's direction is BIT10, Present_Current's is BIT15."""

    class Raw(SimulatedServos):
        def _refresh_present(self, servo):  # keep the registers exactly as the test set them
            pass

    sim = Raw([2])
    bus = ft.FeetechBus(sim)
    sim.servos[2].set(ft.PRESENT_LOAD, 0x400 | 300)  # bit 10 set: -300 (-30 %)
    sim.servos[2].set(ft.PRESENT_CURRENT, 154)
    assert bus.read_load(2) == -300
    assert bus.read_current(2) == 154
    sim.servos[2].set(ft.PRESENT_CURRENT, 0x8000 | 154)  # BIT15 set: a negative current
    assert bus.read_current(2) == -154
    sim.servos[2].set(ft.PRESENT_CURRENT, 0x400 | 154)  # BIT10 is magnitude for current, not sign
    assert bus.read_current(2) == 0x400 | 154
    sim.servos[2].set(ft.PRESENT_LOAD, 1000)
    assert bus.read_load(2) == 1000
    assert ft.PRESENT_LOAD == (60, 2) and ft.PRESENT_CURRENT == (69, 2)  # LeRobot feetech/tables.py


def test_simulated_load_follows_torque_motion_and_stall():
    t = [0.0]
    sim = SimulatedServos([1], clock=lambda: t[0], slew_ticks_per_s=1000)
    bus = ft.FeetechBus(sim)
    assert bus.read_load(1) == 0 and bus.read_current(1) == 0  # torque off: limp

    bus.write(1, ft.TORQUE_ENABLE, 1)
    bus.write(1, ft.GOAL_POSITION, 2048)
    assert 0 < bus.read_load(1) <= 50  # holding still: small load

    bus.write(1, ft.GOAL_POSITION, 2548)  # slewing upward
    assert 100 <= bus.read_load(1) < 500
    bus.write(1, ft.GOAL_POSITION, 1548)  # and downward: the sign flips
    assert -500 < bus.read_load(1) <= -100

    sim.block(1)  # stalled against something while still commanded to move
    t[0] += 5.0
    assert bus.read_load(1) == -1000  # pushing at the torque limit
    assert bus.read_current(1) < -300  # current carries the drive direction too
    bus.write(1, ft.MAX_TORQUE_LIMIT, 500)
    assert bus.read_load(1) == -500  # capped by the configured limit


class _BrokenPort:
    """A transport whose USB-serial link has gone: every call raises like pyserial does."""

    def __init__(self, fail_on=("write", "read", "reset_input_buffer")):
        import serial

        self.exc = serial.SerialException("device reports readiness to read but returned no data")
        self.fail_on = set(fail_on)

    def _maybe_fail(self, op):
        if op in self.fail_on:
            raise self.exc

    def write(self, data):
        self._maybe_fail("write")
        return len(data)

    def read(self, size):
        self._maybe_fail("read")
        return b""

    def reset_input_buffer(self):
        self._maybe_fail("reset_input_buffer")


@pytest.mark.parametrize("fail_on", [("write",), ("read",), ("reset_input_buffer",)])
def test_transport_errors_become_feetech_errors(fail_on):
    """pyserial's SerialException is an OSError, not a FeetechError; the bus must convert it so
    every caller's FeetechError handling (HARDWARE_ERROR, safety events) applies."""
    import serial

    bus = ft.FeetechBus(_BrokenPort(fail_on))
    with pytest.raises(ft.FeetechError, match="transport error") as info:
        bus.read_position(1)
    assert isinstance(info.value.__cause__, serial.SerialException)
    assert bus.ping(1) is False  # ping reports a dead link instead of raising


def test_sync_write_transport_error_becomes_feetech_error():
    bus = ft.FeetechBus(_BrokenPort(("write",)))
    with pytest.raises(ft.FeetechError, match="transport error"):
        bus.sync_write(ft.GOAL_POSITION, {1: 2048, 2: 2048})


class _TermiosBrokenPort(_BrokenPort):
    """pyserial's posix backend lets termios.error escape (it is not an OSError)."""

    def __init__(self, fail_on=("write", "read", "reset_input_buffer")):
        import termios

        super().__init__(fail_on)
        self.exc = termios.error(5, "Input/output error")  # as seen when the USB adapter dropped


@pytest.mark.parametrize("fail_on", [("write",), ("read",), ("reset_input_buffer",)])
def test_termios_errors_become_feetech_errors(fail_on):
    import termios

    bus = ft.FeetechBus(_TermiosBrokenPort(fail_on))
    with pytest.raises(ft.FeetechError, match="transport error") as info:
        bus.read_position(1)
    assert isinstance(info.value.__cause__, termios.error)
    assert bus.ping(1) is False
    with pytest.raises(ft.FeetechError, match="transport error"):
        ft.FeetechBus(_TermiosBrokenPort(("write",))).sync_write(ft.GOAL_POSITION, {1: 2048})
