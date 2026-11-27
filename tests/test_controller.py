from __future__ import annotations

import pytest
import serial

from so_arm101_node import feetech as ft
from so_arm101_node._envelope import CmdRequest
from so_arm101_node.arm import ArmError
from so_arm101_node.node import NS, VERB_TIERS

from .conftest import make_rig

_seq = iter(range(1, 10_000))


def req(verb: str, **params) -> CmdRequest:
    return CmdRequest(f"r{next(_seq)}", verb, params, None, "1.0.0", None)


def run(controller, clock, verb, *, max_s=10.0, events=None, **params):
    """Send one request and tick the fake clock until its response arrives.

    Heartbeats go out every 0.2 s, as the octos bridge sends them for a robot whose
    advert declares a heartbeat timeout.
    """
    request = req(verb, **params)
    response = controller.handle(request)
    waited = 0.0
    while response is None:
        assert waited < max_s, f"{verb} did not finish"
        clock.advance(0.02)
        waited += 0.02
        if round(waited / 0.02) % 10 == 0:
            controller.handle(req("robot.heartbeat"))
        result = controller.tick()
        if events is not None:
            events += result.safety_events
        mine = [r for r in result.responses if r["request_id"] == request.request_id]
        response = mine[0] if mine else None
    return response


def test_motion_requires_torque(rig, clock):
    controller, _ = rig
    r = run(controller, clock, f"{NS}.move_to_named", name="home")
    assert (r["ok"], r["code"]) == (False, "NOT_ENABLED")


def test_move_completes_and_reaches_goal(rig, clock):
    controller, sim = rig
    assert run(controller, clock, f"{NS}.enable_torque")["ok"]
    assert sim.torque(1) and sim.torque(6)
    r = run(controller, clock, f"{NS}.move_to_joint_state",
            joints={"shoulder_pan": 30.0, "elbow_flex": -20.0})
    assert r["ok"], r
    assert r["data"]["joints"]["shoulder_pan"] == pytest.approx(30.0, abs=2.0)
    assert r["data"]["joints"]["elbow_flex"] == pytest.approx(-20.0, abs=2.0)


def test_list_form_and_named_pose(rig, clock):
    controller, _ = rig
    run(controller, clock, f"{NS}.enable_torque")
    r = run(controller, clock, f"{NS}.move_to_joint_state", joints=[10, 0, 0, 0, 0])
    assert r["ok"], r
    r = run(controller, clock, f"{NS}.move_to_named", name="home")
    assert r["ok"] and r["data"]["joints"]["shoulder_pan"] == pytest.approx(0.0, abs=2.0)
    r = run(controller, clock, f"{NS}.move_to_named", name="dance")
    assert r["code"] == "INVALID_PARAMS"


def test_second_motion_while_moving_is_busy(rig, clock):
    controller, _ = rig
    run(controller, clock, f"{NS}.enable_torque")
    assert controller.handle(req(f"{NS}.move_to_joint_state", joints={"shoulder_pan": 40.0})) is None
    r = controller.handle(req(f"{NS}.gripper.open"))
    assert r["code"] == "CONTROLLER_BUSY"


def test_limit_violation(rig, clock):
    controller, _ = rig
    run(controller, clock, f"{NS}.enable_torque")
    r = run(controller, clock, f"{NS}.move_to_joint_state", joints={"elbow_flex": 179.0})
    assert r["code"] == "LIMIT_VIOLATION"


def test_two_arms_need_an_arm_name(clock):
    controller, sim = make_rig(clock, arm_specs=(("left", 1), ("right", 7)))
    r = run(controller, clock, f"{NS}.move_to_named", name="home")
    assert r["code"] == "INVALID_PARAMS" and "left" in r["msg"]
    run(controller, clock, f"{NS}.enable_torque")  # no arm → all arms
    assert sim.torque(1) and sim.torque(12)
    r = run(controller, clock, f"{NS}.move_to_joint_state", arm="right", joints={"shoulder_pan": 20.0})
    assert r["ok"]
    state = controller.state_payload()
    assert state["arms"]["right"]["joints"]["shoulder_pan"] == pytest.approx(20.0, abs=2.0)
    assert state["arms"]["left"]["joints"]["shoulder_pan"] == pytest.approx(0.0, abs=1.0)


def test_stalled_joint_trips_tracking_guard_and_holds(rig, clock):
    controller, sim = rig
    run(controller, clock, f"{NS}.enable_torque")
    sim.block(1)  # shoulder_pan hits something (a pan keeps the gripper height)
    events = []
    r = run(controller, clock, f"{NS}.move_to_joint_state", joints={"shoulder_pan": 60.0}, events=events)
    assert r["code"] == "TRACKING_ERROR"
    assert events and events[0]["kind"] == "tracking_error" and events[0]["joint"] == "shoulder_pan"
    held = r["data"]["joints"]["shoulder_pan"]
    assert sim.goal(1) == pytest.approx(2048 + held * 4095 / 360, abs=2)


def test_gripper_blocked_by_object_still_succeeds(rig, clock):
    controller, sim = rig
    run(controller, clock, f"{NS}.enable_torque")
    run(controller, clock, f"{NS}.gripper.open")
    sim.block(6)  # an object between the jaws
    r = run(controller, clock, f"{NS}.gripper.close")
    assert r["ok"] and r["data"]["gripper_blocked"] is True


def test_joint_state_reports_load_and_a_held_object(rig, clock):
    controller, sim = rig
    state = run(controller, clock, f"{NS}.get_joint_state")["data"]["arms"]["left"]
    assert set(state["load"]) == set(state["joints"]) == set(state["current_ma"])
    assert all(v == 0 for v in state["load"].values())  # torque off

    run(controller, clock, f"{NS}.enable_torque")
    run(controller, clock, f"{NS}.gripper.open")
    free = run(controller, clock, f"{NS}.get_joint_state")["data"]["arms"]["left"]
    assert 0 < free["load"]["gripper"] < 10  # holding open, nothing in the jaws

    sim.block(6)  # an object between the jaws
    assert run(controller, clock, f"{NS}.gripper.close")["data"]["gripper_blocked"] is True
    held = run(controller, clock, f"{NS}.get_joint_state")["data"]["arms"]["left"]
    # configure() caps the gripper at 50 % torque; a held object pushes it to that cap
    assert abs(held["load"]["gripper"]) == pytest.approx(50.0)
    assert abs(held["current_ma"]["gripper"]) > 5 * abs(free["current_ma"]["gripper"])
    assert (held["current_ma"]["gripper"] > 0) == (held["load"]["gripper"] > 0)  # same direction
    assert abs(held["load"]["shoulder_pan"]) < 10  # other joints just hold


def test_negative_current_reaches_get_joint_state_in_ma(rig, clock):
    controller, sim = rig

    # after the simulator's own update, overwrite effort with raw negative readings
    sim._refresh_present = lambda servo, _orig=sim._refresh_present: (
        _orig(servo), servo.set(ft.PRESENT_CURRENT, 0x8000 | 200), servo.set(ft.PRESENT_LOAD, 0x400 | 300))
    state = run(controller, clock, f"{NS}.get_joint_state")["data"]["arms"]["left"]
    assert state["current_ma"]["elbow_flex"] == pytest.approx(-1300.0)  # -200 x 6.5 mA
    assert state["load"]["elbow_flex"] == pytest.approx(-30.0)  # -300 x 0.1 %


def test_effort_timeout_keeps_joint_positions(rig, clock):
    """A servo that stops answering effort reads still reports its position."""
    controller, sim = rig
    expected = run(controller, clock, f"{NS}.get_joint_state")["data"]["arms"]["left"]["joints"]
    effort_registers = {ft.PRESENT_LOAD[0], ft.PRESENT_CURRENT[0]}
    real_write = sim.write

    def write(data):  # servo 3 (elbow_flex) times out on load and current reads only
        if len(data) > 6 and data[2] == 3 and data[4] == ft.INSTR_READ and data[5] in effort_registers:
            return len(data)  # no reply: the bus read times out
        return real_write(data)

    sim.write = write
    r = run(controller, clock, f"{NS}.get_joint_state")
    assert r["ok"], r
    state = r["data"]["arms"]["left"]
    assert state["joints"] == pytest.approx(expected, abs=0.2)
    assert state["load"]["elbow_flex"] is None and state["current_ma"]["elbow_flex"] is None
    assert state["load"]["shoulder_pan"] is not None and state["current_ma"]["gripper"] is not None


class _SerialFailsOnEffort:
    """Wrap the simulated bus so one servo's effort reads raise like a dropped USB-serial port."""

    def __init__(self, sim, servo_id, registers):
        self.sim, self.servo_id, self.registers = sim, servo_id, registers
        self.raised = 0

    def write(self, data):
        if len(data) > 6 and data[2] == self.servo_id and data[4] == ft.INSTR_READ and data[5] in self.registers:
            self.raised += 1
            raise serial.SerialException("device reports readiness to read but returned no data")
        return self.sim.write(data)

    def read(self, size):
        return self.sim.read(size)

    def reset_input_buffer(self):
        self.sim.reset_input_buffer()


@pytest.mark.parametrize("registers, failed", [
    ({ft.PRESENT_CURRENT[0]}, ("current_ma",)),
    ({ft.PRESENT_LOAD[0]}, ("load",)),
    ({ft.PRESENT_LOAD[0], ft.PRESENT_CURRENT[0]}, ("load", "current_ma")),
])
def test_serial_exception_on_an_effort_read_keeps_the_node_answering(rig, clock, registers, failed):
    """A transport error (pyserial SerialException, an OSError) on an optional effort read must not
    escape handle(): positions are kept and only the affected reading becomes null."""
    controller, sim = rig
    expected = run(controller, clock, f"{NS}.get_joint_state")["data"]["arms"]["left"]["joints"]
    flaky = _SerialFailsOnEffort(sim, servo_id=6, registers=registers)  # the gripper servo
    controller.arms["left"].bus._io = flaky
    r = controller.handle(req(f"{NS}.get_joint_state"))  # must return, not raise
    assert flaky.raised > 0
    assert r["ok"], r
    state = r["data"]["arms"]["left"]
    assert state["joints"] == pytest.approx(expected, abs=0.2)  # all six positions still reported
    for field in ("load", "current_ma"):
        assert (state[field]["gripper"] is None) == (field in failed)
        assert state[field]["elbow_flex"] is not None  # other joints unaffected


class _PortDropsAfter:
    """The simulated bus, until ``drop()``: then every call raises pyserial's SerialException."""

    def __init__(self, sim):
        self.sim, self.dropped = sim, False

    def drop(self):
        self.dropped = True

    def _check(self):
        if self.dropped:
            raise serial.SerialException("[Errno 5] Input/output error")

    def write(self, data):
        self._check()
        return self.sim.write(data)

    def read(self, size):
        self._check()
        return self.sim.read(size)

    def reset_input_buffer(self):
        self._check()
        self.sim.reset_input_buffer()


class _TermiosPortDropsAfter(_PortDropsAfter):
    """As _PortDropsAfter, but failing like pyserial's posix backend did on a real USB drop."""

    def _check(self):
        if self.dropped:
            import termios

            raise termios.error(5, "Input/output error")


def test_a_termios_error_mid_hold_does_not_stop_the_node(rig, clock):
    """The real failure: the adapter dropped while the arm was holding after a stopped move;
    reset_input_buffer raised termios.error on the next state read and killed the node."""
    controller, sim = rig
    run(controller, clock, f"{NS}.enable_torque")
    port = _TermiosPortDropsAfter(sim)
    controller.arms["left"].bus._io = port
    port.drop()
    r = controller.handle(req(f"{NS}.get_joint_state"))  # must return, not raise
    assert r["ok"] is False and r["code"] == "HARDWARE_ERROR" and "transport error" in r["msg"]
    controller.state_payload()  # the state stream must not raise either
    clock.advance(0.02)
    controller.tick()  # nor the periodic tick


def test_serial_port_lost_on_a_position_read_is_a_hardware_error(rig, clock):
    controller, sim = rig
    port = _PortDropsAfter(sim)
    controller.arms["left"].bus._io = port
    port.drop()
    r = controller.handle(req(f"{NS}.get_joint_state"))  # must return, not raise
    assert r["ok"] is False and r["code"] == "HARDWARE_ERROR" and "transport error" in r["msg"]
    controller.state_payload()  # the state stream must not raise either


def test_serial_port_lost_during_a_move_ends_it_with_a_hardware_error(rig, clock):
    controller, sim = rig
    run(controller, clock, f"{NS}.enable_torque")
    port = _PortDropsAfter(sim)
    controller.arms["left"].bus._io = port
    request = req(f"{NS}.move_to_joint_state", joints={"shoulder_pan": 40.0})
    assert controller.handle(request) is None
    clock.advance(0.1)
    controller.tick()
    port.drop()
    clock.advance(0.02)
    result = controller.tick()  # must return, not raise
    mine = [r for r in result.responses if r["request_id"] == request.request_id]
    assert mine and mine[0]["code"] == "HARDWARE_ERROR"
    assert any(e["kind"] == "hardware_error" for e in result.safety_events)
    assert controller._motions == {}


def test_position_read_failure_is_still_a_hardware_error(rig, clock):
    controller, sim = rig
    real_write = sim.write
    sim.write = lambda data: (len(data) if len(data) > 6 and data[2] == 3 and data[4] == ft.INSTR_READ
                              and data[5] == ft.PRESENT_POSITION[0] else real_write(data))
    assert run(controller, clock, f"{NS}.get_joint_state")["code"] == "HARDWARE_ERROR"


def test_estop_cancels_motion_and_latches(rig, clock):
    controller, sim = rig
    run(controller, clock, f"{NS}.enable_torque")
    pending = req(f"{NS}.move_to_joint_state", joints={"shoulder_pan": 80.0})
    assert controller.handle(pending) is None
    clock.advance(0.2)
    controller.tick()
    r = controller.handle(req("robot.estop", reason="test"))
    assert r["ok"]
    cancelled = controller.take_cancelled_responses()
    assert [c["code"] for c in cancelled] == ["ESTOPPED"]
    assert cancelled[0]["request_id"] == pending.request_id
    assert sim.torque(1)  # hold, not drop
    r = run(controller, clock, f"{NS}.move_to_named", name="home")
    assert r["code"] == "ESTOPPED"
    assert run(controller, clock, f"{NS}.clear_estop")["ok"]
    assert run(controller, clock, f"{NS}.move_to_named", name="home")["ok"]


def test_estop_torque_off_mode(clock):
    controller, sim = make_rig(clock, estop_action="torque_off")
    run(controller, clock, f"{NS}.enable_torque")
    controller.handle(req("robot.estop"))
    assert not sim.torque(1)


def test_watchdog_holds_when_heartbeats_stop(clock):
    controller, _ = make_rig(clock, heartbeat_timeout_ms=200)
    run(controller, clock, f"{NS}.enable_torque")
    assert controller.handle(req(f"{NS}.move_to_joint_state", joints={"shoulder_pan": 80.0})) is None
    events, responses = [], []
    for _ in range(30):
        clock.advance(0.02)
        result = controller.tick()
        events += result.safety_events
        responses += result.responses
    assert events[0]["kind"] == "watchdog_timeout"
    assert responses[0]["code"] == "ESTOPPED"
    assert controller.estopped


def test_heartbeats_keep_motion_alive(clock):
    controller, _ = make_rig(clock, heartbeat_timeout_ms=200)
    run(controller, clock, f"{NS}.enable_torque")
    request = req(f"{NS}.move_to_joint_state", joints={"shoulder_pan": 50.0})
    assert controller.handle(request) is None
    done = None
    for i in range(500):
        clock.advance(0.02)
        if i % 5 == 0:
            controller.handle(req("robot.heartbeat"))
        result = controller.tick()
        assert not result.safety_events
        if result.responses:
            done = result.responses[0]
            break
    assert done and done["ok"]


def test_disable_torque_requires_rest_or_force(rig, clock):
    controller, sim = rig
    run(controller, clock, f"{NS}.enable_torque")
    r = run(controller, clock, f"{NS}.disable_torque")
    assert r["code"] == "NOT_AT_REST"
    assert sim.torque(1)
    assert run(controller, clock, f"{NS}.move_to_named", name="rest")["ok"]
    assert run(controller, clock, f"{NS}.disable_torque")["ok"]
    assert not sim.torque(1)
    run(controller, clock, f"{NS}.enable_torque")
    assert run(controller, clock, f"{NS}.disable_torque", force=True)["ok"]


def test_bring_up_rejects_missing_servo(clock):
    from so_arm101_node import calibration as calib
    from so_arm101_node.arm import ArmConfig, SoArm101
    from so_arm101_node.feetech import FeetechBus
    from so_arm101_node.node import So101Controller
    from so_arm101_node.sim import SimulatedServos

    sim = SimulatedServos(range(1, 7), clock=clock)
    sim.remove(4)
    config = ArmConfig.from_first_id("arm", 1)
    controller = So101Controller({"arm": SoArm101(FeetechBus(sim), config, calib.default_sim(config.ids))},
                                 clock=clock)
    with pytest.raises(ArmError, match="wrist_flex"):
        controller.bring_up(write_calibration=True)


def test_bring_up_detects_calibration_mismatch(clock):
    controller, sim = make_rig(clock)
    sim.servos[3].set((11, 2), 1234)  # someone rewrote elbow max limit
    with pytest.raises(ArmError, match="recalibrate"):
        controller.bring_up(write_calibration=False)


def test_advert_lists_every_verb_with_a_tier(rig):
    controller, _ = rig
    advert = controller.capabilities_advert()
    verbs = {c["verb"]: c["safety_tier"] for c in advert["commands"]}
    assert verbs == VERB_TIERS
    assert advert["safety"]["heartbeat_timeout_ms"] == 1000
    assert set(verbs.values()) <= {"observe", "safe_motion", "full_actuation", "emergency_override"}
    r = controller.handle(req("vendor.nope"))
    assert r["code"] == "INVALID_PARAMS"


def test_not_reached_holds_where_the_joint_stopped(rig, clock):
    """A joint blocked close to its goal (lag below the tracking limit) must stop pushing."""
    controller, sim = rig
    run(controller, clock, f"{NS}.enable_torque")
    start_encoder = sim.servos[4].encoder
    sim.block(4, start_encoder + 4 * 4095 / 360)  # stops 4° along a 10° move
    events = []
    r = run(controller, clock, f"{NS}.move_to_joint_state", joints={"wrist_flex": 10.0}, events=events)
    assert r["code"] == "NOT_REACHED" and "holding" in r["msg"]
    assert any(e["kind"] == "not_reached" for e in events)
    present_raw = sim.servos[4].present()
    assert abs(sim.goal(4) - present_raw) <= 2  # goal pulled back to where it stopped
