import os
import pickle
from datetime import datetime

from omni.isaac.orbit.assets import ArticulationData


class QuadrupedSimulationData:
    """Class to store the simulation data from the quadruped robot."""

    def __init__(self):
        """Initialize the robot articulation data buffer attributes."""

        self.base_lin_vel_buffer = list()
        self.base_ang_vel_buffer = list()
        self.projected_gravity_buffer = list()
        self.joint_pos_buffer = list()
        self.joint_vel_buffer = list()
        self.joint_acc_buffer = list()
        self.joint_torques_buffer = list()
        self.sim_time_buffer = list()

    def saveStep(self, articulation_data: ArticulationData, sim_time: float = 0.0):
        """Saves the current simulation data in the class attributes."""

        self.base_lin_vel_buffer.append(articulation_data.root_lin_vel_b.clone())
        self.base_ang_vel_buffer.append(articulation_data.root_ang_vel_b.clone())
        self.projected_gravity_buffer.append(articulation_data.projected_gravity_b.clone())
        self.joint_pos_buffer.append(articulation_data.joint_pos.clone() - articulation_data.default_joint_pos.clone())
        self.joint_vel_buffer.append(articulation_data.joint_vel.clone() - articulation_data.default_joint_vel.clone())
        self.joint_acc_buffer.append(articulation_data.joint_acc.clone())
        self.joint_torques_buffer.append(articulation_data.applied_torque.clone())
        self.sim_time_buffer.append(sim_time)

    def saveToFile(self, save_dir):
        """Saves the simulation data to a pickle file."""

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(save_dir, f"{timestamp}.pkl")
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def loadFromFile(cls, filename):
        """Loads the simulation data from a pickle file."""

        with open(filename, "rb") as file:
            return pickle.load(file)
