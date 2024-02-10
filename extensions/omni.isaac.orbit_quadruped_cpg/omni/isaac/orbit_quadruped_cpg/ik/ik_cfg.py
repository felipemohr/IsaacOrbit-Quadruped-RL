from dataclasses import MISSING

from omni.isaac.orbit.utils import configclass


@configclass
class QuadrupedIKCfg:
    """Configuration for the quadruped dimensions used by Inverse Kinematics."""

    hip_length: float = MISSING
    """First segment of the quadruped's lef, the length of the hip."""

    thigh_length: float = MISSING
    """Second segment of the quadruped's lef, the length of the thigh."""

    calf_length: float = MISSING
    """Third segment of the quadruped's lef, the length of the calf."""

    foot_x_offset: float = MISSING
    """Default offset of the foot w.r.t hip reference frame in the x axis."""

    foot_y_offset: float = MISSING
    """Default offset of the foot w.r.t hip reference frame in the y axis."""

    foot_z_offset: float = MISSING
    """Default offset of the foot w.r.t hip reference frame in the z axis."""


GO2_IK_CFG = QuadrupedIKCfg(
    hip_length=0.0955,
    thigh_length=0.213,
    calf_length=0.213,
    foot_x_offset=0.0,
    foot_y_offset=0.0955,
    foot_z_offset=0.301,
)
"""Configuration of Unitree Go2 dimensions used by Inverse Kinematics."""
