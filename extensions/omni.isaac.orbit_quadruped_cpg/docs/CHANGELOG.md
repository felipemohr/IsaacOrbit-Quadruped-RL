# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2024-02-10

### Added

- Add foot offsets parameters in `QuadrupedIKCfg`.
- Add `use_foot_frame` parameter in `get_leg_joints` method.

### Fixed

- Fixed `GO2_TROT_CPG` parameters.

## [0.2.0] - 2024-01-31

### Added

- Created `QuadrupedCPG` and `QuadrupedCPGCfg` classes.


## [0.1.1] - 2024-01-29

### Added

- Created `get_all_leg_joints` method in `QuadrupedIK` class.

### Fixed

- Fixed `GO2_IK_CFG` lengths.
- Fixed `get_leg_joints` method of `QuadrupedIK` return shape.


## [0.1.0] - 2024-01-28

### Added

- Initial release of the extension.
- Created `QuadrupedIK` and `QuadrupedIKCfg` classes.
