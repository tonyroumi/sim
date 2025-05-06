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
        self.model_config_path = os.path.join(self.root_path, "mj_model.json") if config_path is None else config_path
        self.default_config_path = os.path.join(self.root_path, "default_config.json")
        self.xml_path = os.path.join(self.root_path, self.name + ".xml") if xml_path is None else xml_path
        
        with open(self.xml_path, "r") as f:
            self.xml = f.read()

        self.load_robot_config()
        self.initalize()

    def load_robot_config(self):
        """Load the robot's configuration and collision configuration from JSON files.

        Raises:
            FileNotFoundError: If the main configuration file or the collision configuration file does not exist at the specified paths.
        """
        if os.path.exists(self.model_config_path):
            with open(self.model_config_path, "r") as f:
                self.model_config = json.load(f)
        else:
            raise FileNotFoundError(f"No config file found for robot '{self.name}'.")
        
        if os.path.exists(self.default_config_path):
            with open(self.default_config_path, "r") as f:
                self.default_config = json.load(f)
        else:
            raise FileNotFoundError(f"No default config file found for robot '{self.name}'.")

            
    def initalize(self):
        """Initialize the robot's configuration based on the loaded configuration data.
        
        Loads motor ordering, foot names, joint limits, and stores the names of geoms, bodies, and sensors.
        """
    
        self.motor_ordering = list(self.default_config["motors"].keys())
        self.joint_ordering = list(self.default_config["joints"].keys())

        self.nu = len(self.motor_ordering)

        self.default_joint_angles = {}
        self.default_motor_ctrls = {}
        for joint, pos in self.default_config["joints"].items():
            self.default_joint_angles[joint] = pos["default_pos"]
            self.default_motor_ctrls[joint] = self.default_config["motors"][joint]["default_ctrl"]
        

        if "foot_names" in self.default_config["general"]:
            self.foot_names = self.default_config["general"]["foot_names"]

        self.joint_groups = {}
        self.joint_limits = {}
        for joint_name, joint_config in self.default_config["joints"].items():
            self.joint_groups[joint_name] = joint_config["group"]
            self.joint_limits[joint_name] = [joint_config["lower_limit"], joint_config["upper_limit"]]

        self.joints = {}
        self.joint_limits = {}
        for joint in self.model_config["JOINT"].values():
            if joint["name"] == "root":
                continue
            self.joints[joint["name"]] = {
                "qposadr": joint["jnt_qposadr"],
                "jnt_bodyid": joint["jnt_bodyid"],
            }
            self.joint_limits[joint["name"]] = {
                "min": joint["jnt_range"].split(" ")[0],
                "max": joint["jnt_range"].split(" ")[1],
            }

        self.sensors = {}
        for sensor in self.model_config["SENSOR"].values():
            self.sensors[sensor["name"]] = {
                "sensor_type": sensor["sensor_type"],
                "sensor_objtype": sensor["sensor_objtype"],
                "sensor_dim": sensor["sensor_dim"],
                "sensor_adr": sensor["sensor_adr"],
            }

        self.cameras = []
        for camera in self.model_config["CAMERA"].values():
            self.cameras.append(camera["name"])
        
        self.sites = {}
        for site in self.model_config["SITE"].values():
            self.sites[site["name"]] = {
                "site_bodyid": site["site_bodyid"],
                "site_type": site["site_type"],
            }



    def body_to_joint_mapping(self):
        body_data = {int(k.split()[1]): v for k, v in self.model_config["BODY"].items() 
                     if int(k.split()[1]) != 0}
        joint_data = {int(k.split()[1]): v for k, v in self.model_config["JOINT"].items()}

        # Build body id to name and name to id mappings
        body_id_to_name = {bid: binfo['name'] for bid, binfo in body_data.items()}
        body_name_to_id = {v: k for k, v in body_id_to_name.items()}

        # Build parent-child relationship
        body_children = {}
        body_parents = {}
        for bid, binfo in body_data.items():
            pid = int(binfo['body_parentid'])
            body_children.setdefault(pid, []).append(bid)
            body_parents[bid] = pid

        # Map joints to the body they belong to
        joints_by_body = {}
        for jid, jinfo in joint_data.items():
            b_id = int(jinfo['jnt_bodyid'])
            joints_by_body.setdefault(b_id, []).append(jinfo['name'])

        # Recursive function to get all joints under a body (including sub-bodies)
        def collect_all_joints(body_id):
            joints = set(joints_by_body.get(body_id, []))
            for child_id in body_children.get(body_id, []):
                joints.update(collect_all_joints(child_id))
            return joints

        # Final mapping: body name → all joints inside it (including descendants)
        body_to_all_joints = {
            body_id_to_name[bid]: collect_all_joints(bid) for bid in body_id_to_name
        }

        return body_to_all_joints
            