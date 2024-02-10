import math

from omni.isaac.orbit.sim import GroundPlaneCfg, DistantLightCfg
from omni.isaac.orbit.assets import AssetBaseCfg, ArticulationCfg
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.sensors import ContactSensorCfg
from omni.isaac.orbit.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.utils import configclass

from omni.isaac.orbit_quadruped_rl import mdp

from omni.isaac.orbit_assets.unitree import UNITREE_GO2_CFG

##
# Scene definition
##


@configclass
class QuadrupedSceneCfg(InteractiveSceneCfg):
    """Configuration for the quadruped scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=GroundPlaneCfg(size=(100.0, 100.0)))

    # distant light
    light = AssetBaseCfg(
        prim_path="/World/distantLight", spawn=DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0)
    )

    # go2 robot
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.actuators["base_legs"].stiffness = 100.0
    robot.actuators["base_legs"].damping = 1.0

    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=4,
        track_air_time=True,
        update_period=0.0,
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=False,
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        debug_vis=True,
        resampling_time_range=(10.0, 10.0),
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-math.pi / 2, math.pi / 2)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observation for policy group."""

        # observation terms (order preserved)

        # Velocity command
        vel_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # Robot base measurements
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=GaussianNoise(mean=0.0, std=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=GaussianNoise(mean=0.0, std=0.05))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=GaussianNoise(mean=0.0, std=0.025))

        # Robot joints measurements
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=GaussianNoise(mean=0.0, std=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=GaussianNoise(mean=0.0, std=0.5))

        # Feet contact booleans
        feet_contact = ObsTerm(
            func=mdp.feet_contact_bools,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 5.0},
        )

        # Last action
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation group
    policy: PolicyCfg = PolicyCfg()


@configclass
class RandomizationCfg:
    """Configuration for randomization."""

    # startup
    add_base_mass = RandTerm(
        func=mdp.add_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "mass_range": (-1.0, 2.0)},
    )

    # reset
    reset_robot_base = RandTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = RandTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = RandTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    rew_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    rew_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    rew_feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.2,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 0.5,
        },
    )

    pen_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    pen_joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-6)
    pen_joint_powers = RewTerm(func=mdp.joint_powers_l2, weight=-1e-3)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "limit_angle": math.pi / 2},
    )


@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""

    pass


##
# Environment definition
##


@configclass
class QuadrupedEnvCfg(RLTaskEnvCfg):
    """Configuration for the quadruped environment."""

    # Scene settings
    scene: QuadrupedSceneCfg = QuadrupedSceneCfg(num_envs=128, env_spacing=2.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    randomization: RandomizationCfg = RandomizationCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 200.0
