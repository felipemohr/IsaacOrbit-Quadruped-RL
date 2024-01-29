from .ik_cfg import QuadrupedIKCfg

import numpy as np


class QuadrupedIK:
    """Quadruped Inverse Kinematics."""

    def __init__(self, cfg: QuadrupedIKCfg):
        self.cfg = cfg

    def get_leg_joints(self, point: np.ndarray, left=False) -> np.ndarray:
        """Get the joints of a quadruped's leg from a desired (x,y,z) point for the foot w.r.t the hip frame."""

        reflect = 1.0 if left else -1.0

        if point.ndim == 1:
            point = point[:, np.newaxis]
        if point.shape[0] != 3:
            point = point.transpose()

        x, y, z = point[0], point[1], point[2]
        a = np.sqrt(y**2 + z**2 - self.cfg.hip_length**2)
        A = (a**2 + x**2 + self.cfg.thigh_length**2 - self.cfg.calf_length**2) / (
            2 * self.cfg.thigh_length * np.sqrt(a**2 + x**2)
        )
        B = (a**2 + x**2 - self.cfg.thigh_length**2 - self.cfg.calf_length**2) / (
            2 * self.cfg.thigh_length * self.cfg.calf_length
        )

        theta1 = np.arctan2(y, -z) - np.arctan2(reflect * self.cfg.hip_length, a)
        theta2 = np.pi / 2 - np.arctan2(a, x) - np.arctan2(np.sqrt(1.0 - A**2), A)
        theta3 = np.arctan2(np.sqrt(1.0 - B**2), B)

        return np.array([theta1, -theta2, -theta3])
