"""
Direct Overpass API approach for fetching buildings
"""
import json
import requests
from pathlib import Path

def main():
    # Load config
    config_path = Path(__file__).parent.parent / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    map_name = config['current_map']
    bbox = config['maps'][map_name]['bbox']
    
    # Overpass API query
    south, west, north, east = bbox['south'], bbox['west'], bbox['north'], bbox['east']
    
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      way["building"]({south},{west},{north},{east});
      relation["building"]({south},{west},{north},{east});
    );
    out geom;
    """
    
    print(f"Fetching buildings for {map_name} from Overpass API...")
    print(f"Bbox: S={south}, W={west}, N={north}, E={east}")
    
    try:
        response = requests.get(overpass_url, params={'data': overpass_query}, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        print(f"Downloaded {len(data.get('elements', []))} elements")
        
        # Extract building polygons
        building_data = []
        
        for element in data.get('elements', []):
            try:
                if element.get('type') == 'way' and 'geometry' in element:
                    coords = [[node['lon'], node['lat']] for node in element['geometry']]
                    if len(coords) >= 3:
                        building_data.append({
                            'type': 'Polygon',
                            'coordinates': coords
                        })
            except Exception:
                continue
        
        # Save
        output_path = Path(__file__).parent.parent / "data" / f"{map_name}_buildings.json"
        with open(output_path, 'w') as f:
            json.dump(building_data, f, indent=2)
        
        print(f"✅ Saved {len(building_data)} buildings to {output_path}")
        
    except requests.RequestException as e:
        print(f"Error fetching from Overpass API: {e}")
        print("Creating empty buildings file...")
        output_path = Path(__file__).parent.parent / "data" / f"{map_name}_buildings.json"
        with open(output_path, 'w') as f:
            json.dump([], f)

if __name__ == '__main__':
    main()

