# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


## [0.2.3] - 2024-02-22

### Added

- Added `quadruped_simulation_data.py` utils.

### Fixed

- Fixed `joint_powers_l1` penalization computation.

### Changed

- Changed reward weights. 


## [0.2.2] - 2024-02-10

### Changed

- Moved `stiffness` and `damping` changes from `quadruped_env_cfg.py` to standalone `play_cpg.py` script. 


## [0.2.1] - 2024-02-10

### Changed

- Changed `stiffness` and `damping` of the robot in `quadruped_env_cfg.py`. 


## [0.2.0] - 2024-01-21

### Added

- Added new mdp `feet_contact_bools` observation function.
- Added new mdp `joint_powers_l2` reward function.
- Added logs with trained models in 2nd version of the environment.

### Changed

- Changed `RewardsCfg` of `quadruped_env_cfg` to include `joint_powers_l2` penalization instead of `joint_torques_l2`.
- Changed `ObservationsCfg` of `quadruped_env_cfg` to include `feet_contact_bools` observation.


## [0.1.0] - 2024-01-20

### Added

- Initial release of the extension.
- Added `quadruped_env_cfg.py` environment config.
- Added Unitree Go2 RSL_RL agent in `rsl_rl_cfg.py`.
- Added `setup.py` script.
- Added logs directory with first environment version trained model.
