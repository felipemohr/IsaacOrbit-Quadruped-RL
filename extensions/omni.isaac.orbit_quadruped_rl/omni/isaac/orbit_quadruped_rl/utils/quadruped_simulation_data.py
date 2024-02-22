import os
import pickle
import torch
from datetime import datetime

from omni.isaac.orbit.assets import ArticulationData


class QuadrupedSimulationData:
    """Class to store the simulation data from the quadruped robot."""

    def __init__(self, device="cuda:0"):
        """Initialize the robot articulation data buffer attributes."""

        self.base_lin_vel_buffer = torch.empty(0).to(device)
        self.base_ang_vel_buffer = torch.empty(0).to(device)
        self.projected_gravity_buffer = torch.empty(0).to(device)
        self.joint_pos_buffer = torch.empty(0).to(device)
        self.joint_vel_buffer = torch.empty(0).to(device)
        self.joint_acc_buffer = torch.empty(0).to(device)
        self.joint_torques_buffer = torch.empty(0).to(device)
        self.feet_contact_buffer = torch.empty(0).to(device)
        self.sim_time_buffer = torch.empty(0)

    def saveStep(
        self, articulation_data: ArticulationData, feet_contact_bools: torch.Tensor | None = None, sim_time: float = 0.0
    ):
        """Saves the current simulation data in the class attributes."""

        self.base_lin_vel_buffer = torch.cat((self.base_lin_vel_buffer, articulation_data.root_lin_vel_b.clone()))
        self.base_ang_vel_buffer = torch.cat((self.base_ang_vel_buffer, articulation_data.root_ang_vel_b.clone()))
        self.projected_gravity_buffer = torch.cat(
            (self.projected_gravity_buffer, articulation_data.projected_gravity_b.clone())
        )
        self.joint_pos_buffer = torch.cat(
            (self.joint_pos_buffer, articulation_data.joint_pos.clone() - articulation_data.default_joint_pos.clone())
        )
        self.joint_vel_buffer = torch.cat(
            (self.joint_vel_buffer, articulation_data.joint_vel.clone() - articulation_data.default_joint_vel.clone())
        )
        self.joint_acc_buffer = torch.cat((self.joint_acc_buffer, articulation_data.joint_acc.clone()))
        self.joint_torques_buffer = torch.cat((self.joint_torques_buffer, articulation_data.applied_torque.clone()))
        self.sim_time_buffer = torch.cat((self.sim_time_buffer, torch.Tensor([sim_time])))

        if feet_contact_bools is not None:
            self.feet_contact_buffer = torch.cat((self.feet_contact_buffer, feet_contact_bools))

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
