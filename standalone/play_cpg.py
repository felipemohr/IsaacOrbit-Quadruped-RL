import argparse

from omni.isaac.orbit.app import AppLauncher

parser = argparse.ArgumentParser(description="Go2 environment.")
parser.add_argument("--save_data_dir", type=str, default=None, help="Path to save the simulation data.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import traceback

import carb

from omni.isaac.orbit.sim import SimulationContext, SimulationCfg
from omni.isaac.orbit.scene import InteractiveScene
from omni.isaac.orbit.assets import ArticulationCfg

from omni.isaac.orbit_quadruped_rl.quadruped_env_cfg import QuadrupedSceneCfg
from omni.isaac.orbit_quadruped_cpg.cpg import QuadrupedCPG, GO2_TROT_CFG
from omni.isaac.orbit_quadruped_cpg.ik import QuadrupedIK, GO2_IK_CFG

from utils.quadruped_simulation_data import QuadrupedSimulationData


def run_simulator(sim: SimulationContext, scene: InteractiveScene, cpg: QuadrupedCPG, ik: QuadrupedIK):
    """Runs the simulation loop."""

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    render_count = 0

    robot: ArticulationCfg = scene["robot"]
    robot.actuators["base_legs"].stiffness = 100.0
    robot.actuators["base_legs"].damping = 1.0

    with torch.inference_mode():
        while simulation_app.is_running():
            if sim_time >= 30.0 or render_count == 0:
                if args_cli.save_data_dir is not None and render_count > 0:
                    simulation_data.saveToFile(args_cli.save_data_dir)
                simulation_data = QuadrupedSimulationData()
                sim_time = 0.0

                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += scene.env_origins
                robot.write_root_state_to_sim(root_state)

                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)

                action = robot.data.default_joint_pos

                scene.reset()
                cpg.reset()
                print("[INFO]: Resetting robot state...")

            feet_point = cpg.step(sim_dt)
            if sim_time >= 5.0:
                leg_joints = ik.get_all_leg_joints(feet_point[0], feet_point[1], feet_point[2], feet_point[3])
                action = torch.tensor(leg_joints)

            robot.set_joint_position_target(action)
            scene.write_data_to_sim()
            if render_count % 8 == 0:
                sim.step(render=True)
            else:
                sim.step(render=False)
            scene.update(sim_dt)

            sim_time += sim_dt
            render_count += 1

            simulation_data.saveStep(robot.data, sim_time=sim_time)


def main():
    """Main function."""

    sim_cfg = SimulationCfg(dt=1 / 200.0)
    sim = SimulationContext(sim_cfg)

    scene_cfg = QuadrupedSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    cpg = QuadrupedCPG(GO2_TROT_CFG)
    ik = QuadrupedIK(GO2_IK_CFG)

    sim.reset()
    run_simulator(sim, scene, cpg, ik)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log(traceback.format_exc())
        raise
    finally:
        simulation_app.close()
