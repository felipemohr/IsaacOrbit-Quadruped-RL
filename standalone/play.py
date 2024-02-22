import argparse

from omni.isaac.orbit.app import AppLauncher

parser = argparse.ArgumentParser(description="Go2 environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--save_data_dir", type=str, default=None, help="Path to save the simulation data.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from collections import OrderedDict
from torch import nn
import torch
import traceback

import carb

from omni.isaac.orbit.sim import SimulationContext, SimulationCfg
from omni.isaac.orbit.scene import InteractiveScene
from omni.isaac.orbit.managers import SceneEntityCfg

from omni.isaac.orbit_quadruped_rl.quadruped_env_cfg import QuadrupedSceneCfg
from omni.isaac.orbit_quadruped_rl.utils import QuadrupedSimulationData


def load_model(model_path: str) -> nn.Module:
    """Loads the trained model."""

    class ActorNN(nn.Module):
        def __init__(self, num_obs, num_actions, hidden_dims=[128, 128, 128], activation=nn.ELU()):
            super().__init__()

            actor_layers = list()
            actor_layers.append(nn.Linear(num_obs, hidden_dims[0]))
            actor_layers.append(activation)
            for layer_idx, layer_dim in enumerate(hidden_dims):
                if layer_idx == len(hidden_dims) - 1:
                    actor_layers.append(nn.Linear(layer_dim, num_actions))
                else:
                    next_layer_dim = hidden_dims[layer_idx]
                    actor_layers.append(nn.Linear(layer_dim, next_layer_dim))
                    actor_layers.append(activation)
            self.actor = nn.Sequential(*actor_layers)

            print(f"Actor MLP: {self.actor}")

        def forward(self, obs):
            return self.actor(obs)

    model_loaded = torch.load(model_path)
    model = ActorNN(num_obs=52, num_actions=12, hidden_dims=[128, 128, 128], activation=nn.ELU())

    actor_state_dict = OrderedDict(
        (key, value) for key, value in model_loaded["model_state_dict"].items() if "actor" in key
    )
    model.load_state_dict(actor_state_dict)
    model.eval()

    return model


def run_simulator(sim: SimulationContext, scene: InteractiveScene, policy: nn.Module):
    """Runs the simulation loop."""

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    render_count = 0

    robot = scene["robot"]

    feet_contact_threshold = 5.0
    feet_sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*_foot")
    feet_sensor_cfg.resolve(scene)

    kp = 1.0
    ki = 1.0

    with torch.inference_mode():
        while simulation_app.is_running():
            if sim_time >= 30.0 or render_count == 0:
                if args_cli.save_data_dir is not None and render_count > 0:
                    simulation_data.saveToFile(args_cli.save_data_dir)
                    print(f"[INFO]: Saving simulation data to {args_cli.save_data_dir}")
                simulation_data = QuadrupedSimulationData()

                sim_time = 0.0

                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += scene.env_origins
                robot.write_root_state_to_sim(root_state)

                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)

                action = torch.zeros_like(joint_pos)

                sum_vel_error = torch.zeros((args_cli.num_envs, 3)).to(sim.device)

                scene.reset()
                print("[INFO]: Resetting robot state...")

            # Get observations
            # TODO: Read command from keyboard
            # TODO: Markers
            vel_command = torch.stack([torch.Tensor([1.0, 0.0, 0.0])] * args_cli.num_envs).to(sim.device)

            base_lin_vel = robot.data.root_lin_vel_b + 0.05 * torch.randn_like(robot.data.root_lin_vel_b)
            base_ang_vel = robot.data.root_ang_vel_b + 0.05 * torch.randn_like(robot.data.root_ang_vel_b)
            proj_gravity = robot.data.projected_gravity_b + 0.025 * torch.randn_like(robot.data.projected_gravity_b)

            joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
            joint_pos += 0.01 * torch.randn_like(robot.data.joint_pos)
            joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
            joint_vel += 0.5 * torch.randn_like(robot.data.joint_vel)

            feet_net_forces = scene.sensors[feet_sensor_cfg.name].data.net_forces_w
            feet_contact = torch.norm(feet_net_forces[:, feet_sensor_cfg.body_ids], dim=-1) > feet_contact_threshold

            last_action = action

            base_vel = base_lin_vel
            base_vel[:, 2] = base_ang_vel[:, 2]
            vel_error = vel_command - base_vel
            sum_vel_error += vel_error * sim_dt
            sum_vel_error = torch.clamp(sum_vel_error, min=-3.0 / ki, max=3.0 / ki)

            vel_command = kp * vel_error + ki * sum_vel_error
            vel_command = torch.clamp(vel_command, min=-3.0, max=3.0)

            observations = torch.cat(
                (
                    vel_command,
                    base_lin_vel,
                    base_ang_vel,
                    proj_gravity,
                    joint_pos,
                    joint_vel,
                    feet_contact,
                    last_action,
                ),
                dim=1,
            )

            action = policy(observations) * 0.5 + robot.data.default_joint_pos

            robot.set_joint_position_target(action)
            scene.write_data_to_sim()
            if render_count % 8 == 0:
                sim.step(render=True)
            else:
                sim.step(render=False)
            scene.update(sim_dt)

            sim_time += sim_dt
            render_count += 1

            simulation_data.saveStep(robot.data, feet_contact_bools=feet_contact, sim_time=sim_time)


def main():
    """Main function."""

    sim_cfg = SimulationCfg(dt=1 / 200.0)
    sim = SimulationContext(sim_cfg)

    scene_cfg = QuadrupedSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    model = load_model(args_cli.checkpoint).to(sim.device)
    run_simulator(sim, scene, model)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log(traceback.format_exc())
        raise
    finally:
        simulation_app.close()
