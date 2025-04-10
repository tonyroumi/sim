import os
import json
import argparse
import xml.etree.ElementTree as ET

def parse_mujoco_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    config = {
        "general": {},
        "geoms": {},
        "bodies": {},
        "joints": {},
        "sites": {},
        "sensors": {}
    }

    # Extract default attributes for each element type
    defaults = {
        "geom": {},
        "joint": {},
        "site": {},
        "body": {}
    }
    
    # Process defaults recursively
    def extract_defaults(default_elem):
        class_name = default_elem.get("class", "global")
        
        for child in default_elem:
            if child.tag == "default":
                extract_defaults(child)
            
            # Process element types we care about
            elif child.tag in ["geom", "joint", "site", "body"]:
                if class_name not in defaults[child.tag]:
                    defaults[child.tag][class_name] = {}
                defaults[child.tag][class_name].update(child.attrib)
                
                # Process position tags within joint defaults
                if child.tag == "joint":
                    for position in child.findall("./position"):
                        for k, v in position.attrib.items():
                            defaults[child.tag][class_name][f"position_{k}"] = v
            
            # Handle position tag at this level (applies to joints)
            elif child.tag == "position":
                for k, v in child.attrib.items():
                    defaults["joint"][class_name][f"position_{k}"] = v
    
    # Start processing from all default elements
    for default in root.findall(".//default"):
        extract_defaults(default)

    # Extract general simulation options
    option = root.find('option')
    if option is not None:
        config["general"] = {
            "iterations": option.get("iterations"),
            "ls_iterations": option.get("ls_iterations"),
            "timestep": option.get("timestep"),
            "integrator": option.get("integrator"),
        }
        # Remove None values
        config["general"] = {k: v for k, v in config["general"].items() if v is not None}

    # Extract elements and apply defaults
    def extract_elements(element_name, config_key):
        for element in root.findall(f".//{element_name}"):
            element_id = element.get("name")
            if not element_id:
                continue
            
            # Initialize with element's own attributes
            element_attribs = {k: v for k, v in element.attrib.items()}
            
            # For elements that have defaults (geom, joint, site)
            if element_name in defaults:
                # First try to apply "body" defaults if they exist (these are global defaults)
                if "body" in defaults[element_name]:
                    # Start with a copy of body defaults
                    body_defaults = defaults[element_name]["body"].copy()
                    # Then override with element's own attributes
                    body_defaults.update(element_attribs)
                    element_attribs = body_defaults
                
                # Then try to apply global defaults if they exist
                if "global" in defaults[element_name]:
                    # Start with a copy of global defaults
                    global_defaults = defaults[element_name]["global"].copy()
                    # Then override with current attributes
                    global_defaults.update(element_attribs)
                    element_attribs = global_defaults
                
                # Finally apply class-specific defaults if specified
                element_class = element.get("class")
                if element_class and element_class in defaults[element_name]:
                    # Create a new copy with class-specific defaults
                    class_defaults = defaults[element_name][element_class].copy()
                    # Override with current attributes
                    class_defaults.update(element_attribs)
                    element_attribs = class_defaults
            
            # Store the final attributes for this element
            config[config_key][element_id] = element_attribs

    # Extract geoms, bodies, joints, sites
    extract_elements("geom", "geoms")
    extract_elements("body", "bodies")
    extract_elements("joint", "joints")
    extract_elements("site", "sites")

    # Extract sensors
    for sensor in root.findall(".//sensor/*"):  # Find all direct children of sensor
        sensor_name = sensor.get("name")
        sensor_type = sensor.tag
        sensor_attribs = {k: v for k, v in sensor.attrib.items()}
        if sensor_name:
            config["sensors"][sensor_name] = {
                "type": sensor_type,
                "attributes": sensor_attribs
            }
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a config to a robot")
    parser.add_argument("--robot", type=str, help="The name of the robot. Must match the name in robots")
    args = parser.parse_args()

    robot_dir = os.path.join("src", "robots", args.robot)
    robot_xml = os.path.join(robot_dir, args.robot + ".xml")

    cfg = parse_mujoco_xml(robot_xml)

    os.makedirs(robot_dir, exist_ok=True)
    file_path = os.path.join(robot_dir, "config.json")
    with open(file_path, "w") as f:     
        json.dump(cfg, f, indent=4)

    print(f"Config saved to {file_path}")
