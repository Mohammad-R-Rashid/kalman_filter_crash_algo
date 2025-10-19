#!/usr/bin/env python3
"""
Automated map setup script
This script handles the complete workflow of setting up a new map or switching to an existing one.

Usage:
    python setup_map.py                    # Use the map specified in config.json
    python setup_map.py austin             # Switch to austin map
    python setup_map.py wampus             # Switch to wampus map
    python setup_map.py custom             # Setup a custom map (will prompt for bbox)
"""

import sys
import json
import subprocess
from pathlib import Path


def load_config():
    """Load the configuration file"""
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config):
    """Save the configuration file"""
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def add_custom_map(config):
    """Add a custom map to the config"""
    print("\n=== Add Custom Map ===")
    map_name = input("Enter map name: ").strip()
    
    if map_name in config['maps']:
        overwrite = input(f"Map '{map_name}' already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Cancelled.")
            return None
    
    print("\nEnter bounding box coordinates:")
    try:
        north = float(input("  North latitude: ").strip())
        south = float(input("  South latitude: ").strip())
        east = float(input("  East longitude: ").strip())
        west = float(input("  West longitude: ").strip())
    except ValueError:
        print("ERROR: Invalid coordinates. Please enter numeric values.")
        return None
    
    description = input("Description (optional): ").strip() or f"{map_name} area"
    
    config['maps'][map_name] = {
        "bbox": {
            "north": north,
            "south": south,
            "east": east,
            "west": west
        },
        "description": description
    }
    
    save_config(config)
    print(f"✅ Added map '{map_name}' to config")
    return map_name


def run_step(description, command):
    """Run a command step with nice output"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=False)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        return False


def main():
    # Load current config
    config = load_config()
    
    # Determine which map to use
    if len(sys.argv) > 1:
        map_name = sys.argv[1]
        
        if map_name == "custom":
            map_name = add_custom_map(config)
            if not map_name:
                return
        elif map_name not in config['maps']:
            print(f"ERROR: Map '{map_name}' not found in config.")
            print(f"Available maps: {', '.join(config['maps'].keys())}")
            print(f"\nTo add a new map, run: python setup_map.py custom")
            sys.exit(1)
        
        # Update config with new map
        config['current_map'] = map_name
        save_config(config)
        print(f"✅ Switched current map to: {map_name}")
    else:
        map_name = config['current_map']
        print(f"Using map from config: {map_name}")
    
    print(f"\n{'='*60}")
    print(f"Setting up map: {map_name}")
    print(f"Description: {config['maps'][map_name]['description']}")
    print(f"Bounding box: {config['maps'][map_name]['bbox']}")
    print(f"{'='*60}")
    
    # Check if map data already exists
    data_dir = Path(__file__).parent / "data"
    json_file = data_dir / f"{map_name}.json"
    
    # Step 1: Download road network from Overpass API
    if json_file.exists():
        print(f"\n✅ Road network file {json_file} already exists")
        fetch = input("Re-download? (y/n): ").strip().lower()
        if fetch != 'y':
            print("Skipping download")
        else:
            if not run_step("Fetching road network from Overpass API", "python utils/fetch_roads_direct.py"):
                sys.exit(1)
    else:
        if not run_step("Fetching road network from Overpass API", "python utils/fetch_roads_direct.py"):
            sys.exit(1)
    
    # Step 2: Fetch buildings (optional)
    fetch_buildings = input("\nFetch building data? (y/n, recommended): ").strip().lower()
    if fetch_buildings == 'y':
        if not run_step("Fetching building footprints from Overpass API", "python utils/fetch_buildings_direct.py"):
            print("⚠️  Building fetch failed, continuing without buildings")
            # Create empty buildings file
            buildings_file = data_dir / f"{map_name}_buildings.json"
            buildings_file.write_text("[]")
            print(f"Created empty buildings file: {buildings_file}")
    else:
        print("Skipping building fetch")
        # Create empty buildings file
        buildings_file = data_dir / f"{map_name}_buildings.json"
        buildings_file.write_text("[]")
        print(f"Created empty buildings file: {buildings_file}")
    
    print(f"\n{'='*60}")
    print(f"✅ Map setup complete!")
    print(f"{'='*60}")
    print(f"\nYou can now run the simulation:")
    print(f"  python game.py")
    print(f"\nTo switch to a different map in the future:")
    print(f"  python setup_map.py <map_name>")
    print(f"\nAvailable maps: {', '.join(config['maps'].keys())}")


if __name__ == "__main__":
    main()

