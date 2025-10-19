# Map Setup Guide

Complete guide for setting up and switching maps in the collision prediction simulation.

## Overview

The simulation uses OpenStreetMap data fetched via the Overpass API to create realistic road networks and building footprints. All map data is configured through `config.json`.

---

## Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Switch to an existing map
python setup_map.py wampus

# Add and setup a new map
python setup_map.py custom
```

### Option 2: Manual Setup

```bash
# 1. Edit config.json and set your map
# 2. Download roads
python utils/fetch_roads_direct.py

# 3. Download buildings (optional)
python utils/fetch_buildings_direct.py

# 4. Run the simulation
python game.py
```

---

## Configuration File

All maps are defined in `config.json`:

```json
{
  "current_map": "wampus",
  "maps": {
    "wampus": {
      "bbox": {
        "north": 30.29,
        "south": 30.26,
        "east": -97.73,
        "west": -97.75
      },
      "description": "Wampus area"
    }
  }
}
```

### Bounding Box Coordinates

- **north**: Northern latitude boundary (e.g., 30.29)
- **south**: Southern latitude boundary (e.g., 30.26)
- **east**: Eastern longitude boundary (e.g., -97.73)
- **west**: Western longitude boundary (e.g., -97.75)

**Tips for choosing a bounding box:**
- Keep it reasonable: 0.02-0.05° span (~2-5 km) works well
- Too large = slow download and rendering
- Too small = insufficient road network
- Use [OpenStreetMap](https://www.openstreetmap.org) to find coordinates

---

## Adding a New Map

### Method 1: Interactive Setup

```bash
python setup_map.py custom
```

This will prompt you for:
1. Map name
2. Bounding box coordinates (north, south, east, west)
3. Description

### Method 2: Manual Configuration

1. Edit `config.json` and add your map:

```json
{
  "current_map": "my_city",
  "maps": {
    "my_city": {
      "bbox": {
        "north": 40.7589,
        "south": 40.7489,
        "east": -73.9689,
        "west": -73.9889
      },
      "description": "My City Downtown"
    }
  }
}
```

2. Download the map data:

```bash
python utils/fetch_roads_direct.py
python utils/fetch_buildings_direct.py
```

---

## Switching Maps

### Quick Switch

```bash
python setup_map.py <map_name>
```

Example:
```bash
python setup_map.py wampus
python game.py
```

### Manual Switch

Edit `config.json` and change `current_map`:

```json
{
  "current_map": "wampus",
  ...
}
```

Then run:
```bash
python game.py
```

---

## Data Files

For each map named `mymap`, the following files are created in the `data/` directory:

- `mymap.json` - Road network (nodes and edges)
- `mymap_buildings.json` - Building footprints (polygons)

### File Structure

**Roads (`mymap.json`):**
```json
{
  "nodes": {
    "node_id": {
      "x": -97.7457,  // longitude
      "y": 30.2909    // latitude
    }
  },
  "edges": [
    {
      "from": "node1",
      "to": "node2",
      "length_m": 54.5,
      "name": "Main Street",
      "highway": "residential",
      "oneway": false,
      "maxspeed": null,
      "lanes": "2"
    }
  ]
}
```

**Buildings (`mymap_buildings.json`):**
```json
[
  {
    "type": "Polygon",
    "coordinates": [
      [-97.7457, 30.2909],
      [-97.7458, 30.2910],
      [-97.7459, 30.2909],
      [-97.7457, 30.2909]
    ]
  }
]
```

---

## Utility Scripts

### `fetch_roads_direct.py`

Downloads road network from OpenStreetMap via Overpass API.

**What it does:**
- Reads map name from `config.json`
- Fetches all driveable roads in the bounding box
- Filters out footpaths, cycleways, pedestrian paths
- Creates node-edge graph structure
- Saves to `data/<map_name>.json`

**Usage:**
```bash
python utils/fetch_roads_direct.py
```

**Expected output:**
```
Fetching roads for wampus from Overpass API...
Bbox: S=30.26, W=-97.75, N=30.29, E=-97.73
Downloaded 10208 elements
Nodes: 8027, Ways: 2181
Final: 8027 nodes, 14373 edges
✅ Saved to data/wampus.json
```

### `fetch_buildings_direct.py`

Downloads building footprints from OpenStreetMap via Overpass API.

**What it does:**
- Reads map name from `config.json`
- Fetches all buildings in the bounding box
- Extracts polygon coordinates
- Saves to `data/<map_name>_buildings.json`

**Usage:**
```bash
python utils/fetch_buildings_direct.py
```

**Expected output:**
```
Fetching buildings for wampus from OpenStreetMap...
Bbox: S=30.26, W=-97.75, N=30.29, E=-97.73
Downloaded 2407 elements
✅ Saved 2340 buildings to data/wampus_buildings.json
```

---

## Troubleshooting

### Map appears empty or has one road

**Problem:** Incomplete road data download.

**Solution:**
```bash
python utils/fetch_roads_direct.py
python game.py
```

### Buildings not showing

**Problem:** Buildings file missing or empty.

**Solution:**
```bash
python utils/fetch_buildings_direct.py
python game.py
```

### Overpass API timeout

**Problem:** Bounding box too large.

**Solution:** Reduce the area by making the bounding box smaller in `config.json`.

### Wrong geographic area

**Problem:** Coordinates mislabeled or incorrect.

**Solution:** 
1. Check coordinates on [OpenStreetMap](https://www.openstreetmap.org)
2. Verify north > south and east > west
3. Re-download with correct coordinates

---

## Example Workflow

### Setting up a new map for San Francisco

1. Find coordinates on OpenStreetMap:
   - North: 37.8000
   - South: 37.7700
   - East: -122.4000
   - West: -122.4300

2. Run setup script:
```bash
python setup_map.py custom
```

3. Enter details when prompted:
```
Enter map name: san_francisco
  North latitude: 37.8000
  South latitude: 37.7700
  East longitude: -122.4000
  West longitude: -122.4300
Description: San Francisco downtown
```

4. Download data:
```
Fetch building data? (y/n, recommended): y
```

5. Run simulation:
```bash
python game.py
```

---

## Available Maps

Check your current maps:
```bash
python -c "import json; config=json.load(open('config.json')); print('Available maps:', list(config['maps'].keys())); print('Current map:', config['current_map'])"
```

---

## Tips & Best Practices

1. **Start small**: Begin with a 2-3 km² area
2. **Check the area first**: Use OpenStreetMap to verify the area has roads
3. **Buildings are optional**: The simulation works without buildings
4. **Be patient**: Large downloads can take 30-60 seconds
5. **Test incrementally**: Download roads first, verify they work, then add buildings

---

## Overpass API Information

The Overpass API is a read-only API that serves OpenStreetMap data.

**Public endpoint:** `http://overpass-api.de/api/interpreter`

**Rate limits:** 
- Be reasonable with request frequency
- Timeout after 60 seconds for large areas
- If downloads fail, wait a minute and retry

**Alternative instances:**
- https://overpass.kumi.systems/api/interpreter
- https://overpass.openstreetmap.ru/cgi/interpreter

To use an alternative, edit the scripts and change `overpass_url`.

---

## Support

For issues or questions:
1. Check this guide
2. Verify your `config.json` syntax
3. Test with the included `wampus` map first
4. Check the Overpass API status: https://overpass-api.de/api/status

