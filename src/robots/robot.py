import json
import os
from typing import Any, Dict, List
from src.utils.math_utils import degrees_to_radians, radians_to_degrees

class Robot:
    """ This class defines data structure of a robot"""

    def __init__(self, robot_name: str):
        """Initalizes a robot with specified configurations and paths
        
        Args:
            robot_name (str): The name of the robot. Must match the name of the directory in the robots folder
        """
        self.name = robot_name

        self.root_path = os.path.join("src", "robots", robot_name)
        self.config_path = '/home/anthony-roumi/Desktop/sim/src/robots/default_humanoid_legs/config.json'

        with open(self.config_path, "r") as f:
            self.config = json.load(f)

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
        """Initialize the robot's join configurationbased on the generated configuration data.

        Attributes:
            your mom
        """
        self.motor_ordering = []
        for joint_name, joint_config in self.config["joints"].items():
            self.motor_ordering.append(joint_name)
            
        if "foot_name" in self.config["general"]:
            self.foot_name = self.config["general"]["foot_name"]

        self.joint_limits = {}
        for joint_name, joint_config in self.config["joints"].items():
            range = joint_config["range"].split(" ")
            self.joint_limits[joint_name] = [
                degrees_to_radians(float(range[0])),
                degrees_to_radians(float(range[1])),
            ]
        
        
        
        

            
        
