import os
import json
import argparse
import xml.etree.ElementTree as ET
import mujoco
import tempfile

def print_mj_model(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)

    with tempfile.NamedTemporaryFile(mode='r+', delete=True) as tmpfile:
    # tmpfile.name is the path to pass to the C function
        mujoco.mj_printModel(model, tmpfile.name)
        tmpfile.seek(0)
        mj_model_str = tmpfile.read()
    return mj_model_str

def parse_mj_model(xml_path: str):
    relevant_sections = set(['BODY', 'JOINT', 'DOF', 'GEOM', 'SITE', 'CAMERA', 'ACTUATOR', 'SENSOR'])
    result = {section: {} for section in relevant_sections}
    
    # Add fields for default joint angles and motor controls
    result['default_joint_angles'] = {}
    result['default_motor_ctrls'] = {}

    mj_model_str = print_mj_model(xml_path)
    lines = mj_model_str.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue

        parts = line.split()
        if not parts:
            i += 1
            continue
            
        current_section = parts[0]
        
        # Handle key_qpos0 (default joint angles)
        if current_section == 'key_qpos0':
            # Skip the first element which is 'key_qpos0'
            result['default_joint_angles'] = [float(x) for x in parts[1:]]
            i += 1
        # Handle key_ctrl0 (default motor controls)
        elif current_section == 'key_ctrl0':
            # Skip the first element which is 'key_ctrl0'
            result['default_motor_ctrls'] = [float(x) for x in parts[1:]]
            i += 1
        # Handle regular sections we're interested in
        elif current_section in relevant_sections:
            result[current_section][line[:-1]] = {}
            i += 1

            # Iterate through the section's lines
            while i < len(lines) and lines[i] and lines[i].split()[0] not in relevant_sections and lines[i].split()[0] not in ['key_qpos0', 'key_ctrl0']:
                section_name = lines[i].split()[0]
                result[current_section][line[:-1]][section_name] = ' '.join(lines[i].split()[1:])
                i += 1
        else:
            i += 1

    return result

def get_default_config(mj_model: dict):
    config = {}
    general = {}
    motor_config = {}

    foot_names = []
    
    # Find foot bodies
    for body_key, body_data in mj_model["BODY"].items():
        if "name" in body_data and "foot" in body_data["name"].lower():
            foot_names.append(body_data["name"])
    
    # Add foot names to general config
    if foot_names:
        general["foot_names"] = foot_names
    default_motor_ctrls = mj_model.get("default_motor_ctrls", [])
    # Create a mapping from joint IDs to DOF properties
    joint_to_dof_map = {}
    ctrl_index = 0

    for actuator_key, actuator_data in mj_model["ACTUATOR"].items():
        if "name" in actuator_data:
            actuator_name = actuator_data["name"]
            
            # Get initial control value
            default_ctrl = 0.0
            if ctrl_index < len(default_motor_ctrls):
                default_ctrl = float(default_motor_ctrls[ctrl_index])
            
            # Get kp (proportional gain) if available
            kp = 0.0
            if "actuator_gainprm" in actuator_data:
                # Typically the first value in gainprm is kp
                gain_values = actuator_data["actuator_gainprm"].split()
                if len(gain_values) > 0:
                    kp = float(gain_values[0])
            
            # Store configuration for this motor
            motor_config[actuator_name] = {
                "default_ctrl": default_ctrl,
                "kp": kp
            }
            
            # Track the joint this actuator controls if available
            if "actuator_trnid" in actuator_data:
                trnid_values = actuator_data["actuator_trnid"].split()
                if len(trnid_values) > 0:
                    joint_id = int(trnid_values[0])
                    motor_config[actuator_name]["joint_id"] = joint_id
            
            ctrl_index += 1
    
    
    # Process DOF section to map joint IDs to their properties
    for dof_key, dof_data in mj_model["DOF"].items():
        if "dof_jntid" in dof_data:
            joint_id = int(dof_data["dof_jntid"])
            
            # Extract DOF properties
            armature = float(dof_data.get("dof_armature", "0"))
            damping = float(dof_data.get("dof_damping", "0"))
            frictionloss = float(dof_data.get("dof_frictionloss", "0"))
            
            # Store properties mapped to joint ID
            joint_to_dof_map[joint_id] = {
                "armature": armature,
                "damping": damping,
                "frictionloss": frictionloss
            }
    
    # Get default joint angles for init_pos
    default_joint_angles = mj_model.get("default_joint_angles", [])
    
    # Initialize position index to 0, will be updated to 7 if a root joint is found
    qpos_start_idx = 0
    
    # Iterate through all joints in the model
    for joint_id, joint_data in mj_model["JOINT"].items():
        joint_id = joint_id.split()[-1]
        joint_name = joint_data["name"] 
        
        print("Available groups: none (root), neck, waist, arm, leg")
        group = input(f"Which group does joint '{joint_name}' belong to? ").strip().lower()
        
        # Validate input - ensure it's one of the allowed groups
        while group not in ["none","neck", "waist", "arm", "leg"]:
            print("Invalid group. Please choose from: none (root), neck, waist, arm, leg")
            group = input(f"Which group does joint '{joint_name}' belong to? ").strip().lower()
        
        # Extract joint limits from the model
        lower_limit = float(joint_data.get("jnt_range", "0 0").split()[0])
        upper_limit = float(joint_data.get("jnt_range", "0 0").split()[1])
        
        # Extract joint stiffness if available, otherwise use default
        stiffness = float(joint_data.get("jnt_stiffness", "0"))
        
        # Parse the joint_id for numeric identification
        numeric_joint_id = int(joint_id.split()[-1])
        
        # Determine the starting index for joint positions based on root joint
        if group == "none":
            qpos_start_idx = 7
            general["is_fixed"] = False
            continue  # Skip adding root joints to config
        
        # Get the initial position for this joint if available
        default_pos = 0.0
        current_pos_idx = qpos_start_idx + (numeric_joint_id - (1 if qpos_start_idx == 7 else 0))
        if current_pos_idx < len(default_joint_angles):
            default_pos = default_joint_angles[current_pos_idx]
        
        # Create joint configuration with basic properties
        config[joint_name] = {
            "group": group,
            "lower_limit": lower_limit,
            "upper_limit": upper_limit,
            "stiffness": stiffness,
            "id": joint_id,
            "default_pos": default_pos
        }
        
        # Add DOF properties if they exist for this joint
        if numeric_joint_id in joint_to_dof_map:
            dof_props = joint_to_dof_map[numeric_joint_id]
            config[joint_name]["armature"] = dof_props["armature"]
            config[joint_name]["damping"] = dof_props["damping"]
            config[joint_name]["frictionloss"] = dof_props["frictionloss"]
        else:
            # Set defaults if no DOF properties were found
            config[joint_name]["armature"] = 0.0
            config[joint_name]["damping"] = 0.0
            config[joint_name]["frictionloss"] = 0.0
    
    # Combine joint config with general config
    result = {
        "general": general,
        "joints": config,
        "motors": motor_config
    }
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a config to a robot")
    parser.add_argument("--robot", type=str, required=True, help="The name of the robot. Must match the name in robots")
    args = parser.parse_args()

    robot_dir = os.path.join("src", "robots", args.robot)
    robot_xml = os.path.join(robot_dir, args.robot + ".xml")
    
    #Get mujoco model
    mj_model = parse_mj_model(robot_xml)

    os.makedirs(robot_dir, exist_ok=True)
    file_path = os.path.join(robot_dir, "mj_model.json")
    with open(file_path, "w") as f:     
        json.dump(mj_model, f, indent=4)
    
    default_config = get_default_config(mj_model)
    with open(os.path.join(robot_dir, "default_config.json"), "w") as f:
        json.dump(default_config, f, indent=4)

    print(f"Mujoco model saved to {file_path}")