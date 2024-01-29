"""Installation script for the 'omni.isaac.orbit_quadruped_cpg' python package."""

import os
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "numpy",
    "torch",
    "protobuf>=3.20.2",
]

# Installation operation
setup(
    name="omni-isaac-orbit_quadruped_cpg",
    author="Felipe Mohr",
    maintainer="Felipe Mohr",
    maintainer_email="felipe18mohr@gmail.com",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=INSTALL_REQUIRES,
    packages=["omni.isaac.orbit_quadruped_cpg"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 2023.1.0-hotfix.1",
    ],
    zip_safe=False,
)
