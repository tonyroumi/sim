import json
import os
from typing import Any, Dict, List
from src.utils.math_utils import degrees_to_radians, radians_to_degrees

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

class Robot:
    """ This class defines data structure of a robot"""

    def __init__(self, robot_name: str, config_path: str = None, xml_path: str = None):
        """Initalizes a robot with specified configurations and paths
        
        Args:
            robot_name (str): The name of the robot. Must match the name of the directory in the robots folder
        """
        self.name = robot_name

        self.root_path = os.path.join(project_root, "src", "robots", robot_name)
        self.config_path = os.path.join(self.root_path, "config.json") if config_path is None else config_path
        self.xml_path = os.path.join(self.root_path, self.name + ".xml") if xml_path is None else xml_path

        with open(self.config_path, "r") as f:
            self.config = json.load(f)
        
        with open(self.xml_path, "r") as f:
            self.xml = f.read()

        self.load_robot_config()
        self.initalize()

    def load_robot_config(self):
        """Load the robot's configuration and collision configuration from JSON files.

        Raises:
            FileNotFoundError: If the main configuration file or the collision configuration file does not exist at the specified paths.
        """
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                self.config = json.load(f)

        else:
            raise FileNotFoundError(f"No config file found for robot '{self.name}'.")

            
    def initalize(self):
        """Initialize the robot's configuration based on the loaded configuration data.
        
        Loads motor ordering, foot names, joint limits, and stores the names of geoms, bodies, and sensors.
        """
        # Load motor ordering
        self.motor_ordering = []
        for joint_name, joint_config in self.config["joints"].items():
            self.motor_ordering.append(joint_name)
            
        # Load foot name if specified
        if "foot_name" in self.config["general"]:
            self.foot_name = self.config["general"]["foot_name"]

        # Initialize joint limits dictionary
        self.joint_limits = {}
        for joint_name, joint_config in self.config["joints"].items():
            if "range" in joint_config:
                range_str = joint_config["range"]
                min_val, max_val = map(float, range_str.split())
                self.joint_limits[joint_name] = {"min": min_val, "max": max_val}
        
        # Store geom names
        self.geom_names = []
        if "geoms" in self.config:
            self.geom_names = list(self.config["geoms"].keys())
        
        # Store body names
        self.body_names = []
        if "bodies" in self.config:
            self.body_names = list(self.config["bodies"].keys())
        
        # Store sensor names
        self.sensor_names = []
        if "sensors" in self.config:
            self.sensor_names = list(self.config["sensors"].keys())

            
