import argparse
import os

from omni.isaac.orbit.app import AppLauncher

parser = argparse.ArgumentParser(description="Gait data.")
parser.add_argument("--sim_data_file", type=str, default=None, help="Path to simulation data.")
parser.add_argument("--save_plot_path", type=str, default=None, help="Path to save the plot.")
parser.add_argument("--avg_window_size", type=int, default=1, help="Size of moving average window size.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from omni.isaac.orbit_quadruped_rl.utils import QuadrupedSimulationData
import matplotlib.pyplot as plt
import numpy as np

sim_data_policy: QuadrupedSimulationData = QuadrupedSimulationData.loadFromFile(args_cli.sim_data_file)

sim_time_policy = sim_data_policy.sim_time_buffer.to("cpu").numpy()
feet_contact_policy = sim_data_policy.feet_contact_buffer.to("cpu").numpy()

FL_contact_avg = (
    np.convolve(feet_contact_policy[:, 0], np.ones(args_cli.avg_window_size) / args_cli.avg_window_size, mode="valid")
    > 0.5
)
FR_contact_avg = (
    np.convolve(feet_contact_policy[:, 1], np.ones(args_cli.avg_window_size) / args_cli.avg_window_size, mode="valid")
    > 0.5
)
RL_contact_avg = (
    np.convolve(feet_contact_policy[:, 2], np.ones(args_cli.avg_window_size) / args_cli.avg_window_size, mode="valid")
    > 0.5
)
RR_contact_avg = (
    np.convolve(feet_contact_policy[:, 3], np.ones(args_cli.avg_window_size) / args_cli.avg_window_size, mode="valid")
    > 0.5
)


# Plot feet contact
plt.figure(figsize=(8, 4))
plt.plot(sim_time_policy[2000:2500], 4.5 + FL_contact_avg[2000:2500], label="FL")
plt.plot(sim_time_policy[2000:2500], 3.0 + FR_contact_avg[2000:2500], label="FR")
plt.plot(sim_time_policy[2000:2500], 1.5 + RL_contact_avg[2000:2500], label="RL")
plt.plot(sim_time_policy[2000:2500], 0.0 + RR_contact_avg[2000:2500], label="RR")
plt.legend(fontsize=14)
plt.xlabel("Time (s)", fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel("Contact", fontsize=14)
plt.yticks([], fontsize=14)
plt.ylim(-0.5, 6.0)
plt.grid(True)
if args_cli.save_plot_path is not None:
    plt.savefig(os.path.join(args_cli.save_plot_path, "gait.png"), bbox_inches="tight", dpi=300)
plt.show()
