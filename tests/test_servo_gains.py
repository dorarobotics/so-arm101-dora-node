"""Servo position gains written by ``SoArm101.configure()`` at bring-up.

LeRobot's SO-101 follower uses P = 16 on every servo (less jitter). Under gravity that
leaves the elbow and the shoulder lift 2-3 degrees off their goals on real arms, outside
the node's 2 degree settle tolerance, so those two joints use the servo default P = 32.
The other gains are unchanged.
"""

from __future__ import annotations

from so_arm101_node import feetech as ft

from .conftest import make_rig

JOINT_IDS = {"shoulder_pan": 1, "shoulder_lift": 2, "elbow_flex": 3, "wrist_flex": 4, "wrist_roll": 5, "gripper": 6}


def last_written(sim, register) -> dict[str, int]:
    address = register[0]
    by_id = {sid: value for sid, addr, value in sim.writes if addr == address}
    return {joint: by_id[sid] for joint, sid in JOINT_IDS.items()}


def test_shoulder_lift_and_elbow_use_the_servo_default_p_gain(clock):
    _, sim = make_rig(clock)
    assert last_written(sim, ft.P_COEFFICIENT) == {
        "shoulder_pan": 16,
        "shoulder_lift": 32,
        "elbow_flex": 32,
        "wrist_flex": 16,
        "wrist_roll": 16,
        "gripper": 16,
    }


def test_i_and_d_gains_are_unchanged(clock):
    _, sim = make_rig(clock)
    assert set(last_written(sim, ft.I_COEFFICIENT).values()) == {0}
    assert set(last_written(sim, ft.D_COEFFICIENT).values()) == {32}
