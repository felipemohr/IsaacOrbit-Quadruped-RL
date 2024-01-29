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


GO2_IK_CFG = QuadrupedIKCfg(hip_length=0.1, thigh_length=0.2, calf_length=0.2)
