"""
Direct Overpass API approach for fetching roads
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
    
    # Overpass API query for roads
    south, west, north, east = bbox['south'], bbox['west'], bbox['north'], bbox['east']
    
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      way["highway"]["highway"!~"footway|path|cycleway|steps|pedestrian"]({south},{west},{north},{east});
    );
    out body;
    >;
    out skel qt;
    """
    
    print(f"Fetching roads for {map_name} from Overpass API...")
    print(f"Bbox: S={south}, W={west}, N={north}, E={east}")
    
    try:
        response = requests.get(overpass_url, params={'data': overpass_query}, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        print(f"Downloaded {len(data.get('elements', []))} elements")
        
        # Build node dictionary
        nodes = {}
        ways = []
        
        for element in data.get('elements', []):
            if element.get('type') == 'node':
                node_id = str(element['id'])
                nodes[node_id] = {
                    'x': element['lon'],
                    'y': element['lat']
                }
            elif element.get('type') == 'way':
                ways.append(element)
        
        print(f"Nodes: {len(nodes)}, Ways: {len(ways)}")
        
        # Build edges from ways
        edges = []
        for way in ways:
            way_nodes = [str(n) for n in way.get('nodes', [])]
            tags = way.get('tags', {})
            highway = tags.get('highway', 'unknown')
            name = tags.get('name', 'unknown')
            oneway = tags.get('oneway', 'no') in ['yes', 'true', '1']
            maxspeed = tags.get('maxspeed')
            lanes = tags.get('lanes')
            
            # Create edges between consecutive nodes
            for i in range(len(way_nodes) - 1):
                from_node = way_nodes[i]
                to_node = way_nodes[i + 1]
                
                if from_node in nodes and to_node in nodes:
                    # Calculate length
                    from_coords = nodes[from_node]
                    to_coords = nodes[to_node]
                    dx = (to_coords['x'] - from_coords['x']) * 111320 * 0.87  # approx meters
                    dy = (to_coords['y'] - from_coords['y']) * 110540  # approx meters
                    length = (dx**2 + dy**2) ** 0.5
                    
                    edges.append({
                        'from': from_node,
                        'to': to_node,
                        'length_m': length,
                        'name': name,
                        'highway': highway,
                        'oneway': oneway,
                        'maxspeed': maxspeed,
                        'lanes': lanes
                    })
                    
                    # Add reverse edge if not oneway
                    if not oneway:
                        edges.append({
                            'from': to_node,
                            'to': from_node,
                            'length_m': length,
                            'name': name,
                            'highway': highway,
                            'oneway': False,
                            'maxspeed': maxspeed,
                            'lanes': lanes
                        })
        
        # Filter nodes to only those used in edges
        used_nodes = set()
        for edge in edges:
            used_nodes.add(edge['from'])
            used_nodes.add(edge['to'])
        
        filtered_nodes = {nid: nodes[nid] for nid in used_nodes}
        
        print(f"Final: {len(filtered_nodes)} nodes, {len(edges)} edges")
        
        # Create output JSON
        output_data = {
            'nodes': filtered_nodes,
            'edges': edges
        }
        
        # Save
        output_path = Path(__file__).parent.parent / "data" / f"{map_name}.json"
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✅ Saved to {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

