from .cpg_cfg import QuadrupedCPGCfg

import numpy as np


class QuadrupedCPG:
    """Quadruped Central Pattern Generator."""

    def __init__(self, cfg: QuadrupedCPGCfg):
        # TODO: include number of environments
        self.cfg = cfg

        self.dstep_x = 0.1 * np.ones(4)
        self.dstep_y = 0.00 * np.ones(4)

        self.reset()

    def reset(self):
        """Reset the Central Pattern Generator."""
        self._frequency_omega = self.cfg.stance_frequency * np.ones(4)
        self._ground_multiplier = np.zeros(4)

        self._amplitude_r = np.random.rand(4)
        self._amplitude_dr = np.zeros(4)
        self._amplitude_d2r = np.zeros(4)

        self._phase_theta = np.random.rand(4)
        self._phase_dtheta = np.zeros(4)

    def step(self, dt: float):
        """Steps the CPG algorithm providing xyz feet positions."""

        self._amplitude_d2r = self.cfg.convergence_factor * (
            self.cfg.convergence_factor / 4.0 * (self.cfg.amplitude_mu - self._amplitude_r) - self._amplitude_dr
        )
        self._amplitude_dr += self._amplitude_d2r * dt

        for i in range(4):
            self._frequency_omega[i] = (
                2 * np.pi * self.cfg.swing_frequency
                if self._phase_theta[i] < np.pi
                else 2 * np.pi * self.cfg.stance_frequency
            )
            self._phase_dtheta[i] = self._frequency_omega[i]
            for j in range(4):
                self._phase_dtheta[i] += (
                    self._amplitude_r[j]
                    * self.cfg.coupling_weight
                    * np.sin(self._phase_theta[j] - self._phase_theta[i] - self.cfg.coupling_matrix[i][j])
                )

        self._amplitude_r += self._amplitude_dr * dt
        self._phase_theta += self._phase_dtheta * dt

        self._phase_theta %= 2 * np.pi
        self._ground_multiplier = np.where(
            np.sin(self._phase_theta) > 0, self.cfg.ground_clearance, self.cfg.ground_penetration
        )

        foot_x = -self.dstep_x * self._amplitude_r * np.cos(self._phase_theta)
        foot_y = -self.dstep_y * self._amplitude_r * np.cos(self._phase_theta)
        foot_z = self._ground_multiplier * np.sin(self._phase_theta)

        return np.array([foot_x, foot_y, foot_z]).transpose()
    