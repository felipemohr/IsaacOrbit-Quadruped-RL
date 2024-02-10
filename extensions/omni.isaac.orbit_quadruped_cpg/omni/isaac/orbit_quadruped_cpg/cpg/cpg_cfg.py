import numpy as np

from dataclasses import MISSING

from omni.isaac.orbit.utils import configclass


@configclass
class QuadrupedCPGCfg:
    """Configuration class for the quadruped Central Pattern Generator parameters."""

    convergence_factor: float = 50.0
    """Convergence factor used by the neurons in the CPG network."""

    coupling_weight: float = 1.0
    """The coupling weight of connections between the neurons in the CPG network."""

    amplitude_mu: float = 1.0
    """Amplitude used by the CPG."""

    coupling_matrix: np.ndarray = MISSING
    """Coupling matrix of the neurons that defines the gait pattern of the CPG network."""

    swing_frequency: float = MISSING
    """Frequency of the legs when in swing phase, in Hertz."""

    stance_frequency: float = MISSING
    """Frequency of the legs when in stance phase, in Hertz."""

    ground_clearance: float = MISSING
    """Distance between the foot and the ground when at the height of the swing phase."""

    ground_penetration: float = MISSING
    """How far the foot must penetrate on the ground when in stance phase."""


##
# Coupling matrices
##

walk_gait_matrix = np.array(
    [
        [0, np.pi, np.pi / 2, 3 * np.pi / 2],
        [-np.pi, 0, -np.pi / 2, -3 * np.pi / 2],
        [-np.pi / 2, np.pi / 2, 0, -np.pi],
        [-3 * np.pi / 2, 3 * np.pi / 2, np.pi, 0],
    ]
)

trot_gait_matrix = np.array(
    [[0, np.pi, np.pi, 0], [-np.pi, 0, 0, -np.pi], [-np.pi, 0, 0, -np.pi], [0, np.pi, np.pi, 0]]
)

pace_gait_matrix = np.array(
    [[0, np.pi, np.pi, np.pi], [-np.pi, 0, -np.pi, 0], [0, np.pi, 0, np.pi], [-np.pi, 0, -np.pi, 0]]
)

gallop_gait_matrix = np.array(
    [[0, 0, -np.pi, -np.pi], [0, 0, -np.pi, -np.pi], [np.pi, np.pi, 0, 0], [np.pi, np.pi, 0, 0]]
)

##
# Configuration
##

GO2_TROT_CFG = QuadrupedCPGCfg(
    coupling_matrix=trot_gait_matrix,
    swing_frequency=2.5,
    stance_frequency=1.5,
    ground_clearance=0.05,
    ground_penetration=0.005,
)
"""Configuration for CPG locomotion of Unitree Go2 using trot gait."""
