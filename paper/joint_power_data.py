import argparse
import os

from omni.isaac.orbit.app import AppLauncher

parser = argparse.ArgumentParser(description="Joint Power data.")
parser.add_argument("--sim_data_file_cpg", type=str, default=None, help="Path to CPG simulation data.")
parser.add_argument("--sim_data_file_policy", type=str, default=None, help="Path to policy simulation data.")
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

sim_data_cpg: QuadrupedSimulationData = QuadrupedSimulationData.loadFromFile(args_cli.sim_data_file_cpg)
sim_data_policy: QuadrupedSimulationData = QuadrupedSimulationData.loadFromFile(args_cli.sim_data_file_policy)

# Read CPG data
base_lin_vel_cpg = sim_data_cpg.base_lin_vel_buffer.to("cpu").numpy()
sim_time_cpg = sim_data_cpg.sim_time_buffer.to("cpu").numpy()
joint_pos_cpg = sim_data_cpg.joint_pos_buffer.to("cpu").numpy()
joint_vel_cpg = sim_data_cpg.joint_vel_buffer.to("cpu").numpy()
joint_acc_cpg = sim_data_cpg.joint_acc_buffer.to("cpu").numpy()
joint_torques_cpg = sim_data_cpg.joint_torques_buffer.to("cpu").numpy()

# Read Trained Policy data
base_lin_vel_policy = sim_data_policy.base_lin_vel_buffer.to("cpu").numpy()
sim_time_policy = sim_data_policy.sim_time_buffer.to("cpu").numpy()
joint_pos_policy = sim_data_policy.joint_pos_buffer.to("cpu").numpy()
joint_vel_policy = sim_data_policy.joint_vel_buffer.to("cpu").numpy()
joint_acc_policy = sim_data_policy.joint_acc_buffer.to("cpu").numpy()
joint_torques_policy = sim_data_policy.joint_torques_buffer.to("cpu").numpy()

# Joint powers
joint_power_cpg = np.sum(np.abs(joint_torques_cpg * joint_vel_cpg), axis=1)
joint_power_policy = np.sum(np.abs(joint_torques_policy * joint_vel_policy), axis=1)

joint_power_cpg_avg = np.convolve(
    joint_power_cpg, np.ones(args_cli.avg_window_size) / args_cli.avg_window_size, mode="valid"
)
joint_power_policy_avg = np.convolve(
    joint_power_policy, np.ones(args_cli.avg_window_size) / args_cli.avg_window_size, mode="valid"
)


# Plot joint powers
plt.figure(figsize=(8, 4))
plt.plot(sim_time_cpg[2000:4000], joint_power_cpg[2000:4000], alpha=0.18, color="blue")
plt.plot(sim_time_cpg[2000:4000], joint_power_cpg_avg[2000:4000], label="CPG", color="blue")
plt.plot(sim_time_policy[2000:4000], joint_power_policy[2000:4000], alpha=0.18, color="red")
plt.plot(sim_time_policy[2000:4000], joint_power_policy_avg[2000:4000], label="Policy", color="red")
plt.legend(fontsize=14)
plt.xlabel("Time (s)", fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel("Joint Power (W)", fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0, 150)
plt.grid(True)
if args_cli.save_plot_path is not None:
    plt.savefig(os.path.join(args_cli.save_plot_path, "power_consumption.png"), bbox_inches="tight", dpi=300)
plt.show()

m = 15.0
g = 9.81

mean_vel_cpg = np.mean(base_lin_vel_cpg[2000:4000, 0])
mean_power_cpg = np.mean(joint_power_cpg[2000:4000])
CoT_cpg = mean_power_cpg / (m * g * mean_vel_cpg)

mean_vel_policy = np.mean(base_lin_vel_policy[2000:4000, 0])
mean_power_policy = np.mean(joint_power_policy[2000:4000])
CoT_policy = mean_power_policy / (m * g * mean_vel_policy)

print("CPG:")
print("mean velocity", mean_vel_cpg)
print("mean joint power: ", mean_power_cpg)
print("CoT", CoT_cpg)
print()
print("Policy:")
print("mean velocity", mean_vel_policy)
print("mean joint power: ", mean_power_policy)
print("CoT", CoT_policy)
