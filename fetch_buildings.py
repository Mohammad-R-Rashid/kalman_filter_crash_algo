"""
Fetch building footprints from OSM to add context to the simulation
"""
import osmnx as ox
import json

def main():
    # Same bbox as the road network
    bbox = (30.29, 30.26, -97.73, -97.75)  # (north, south, east, west)
    
    print("Fetching building footprints from OpenStreetMap...")
    
    try:
        # Get buildings as GeoDataFrames
        tags = {'building': True}
        buildings = ox.features_from_bbox(bbox, tags=tags)
        
        print(f"Found {len(buildings)} buildings")
        
        # Extract building polygons
        building_data = []
        for idx, building in buildings.iterrows():
            geom = building.geometry
            
            # Handle different geometry types
            if geom.geom_type == 'Polygon':
                coords = list(geom.exterior.coords)
                building_data.append({
                    'type': 'Polygon',
                    'coordinates': [[lon, lat] for lon, lat in coords]
                })
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    coords = list(poly.exterior.coords)
                    building_data.append({
                        'type': 'Polygon',
                        'coordinates': [[lon, lat] for lon, lat in coords]
                    })
        
        # Save to JSON
        with open('buildings.json', 'w') as f:
            json.dump(building_data, f, indent=2)
        
        print(f"Saved {len(building_data)} building polygons to buildings.json")
        
    except Exception as e:
        print(f"Error fetching buildings: {e}")
        print("Creating empty buildings file...")
        with open('buildings.json', 'w') as f:
            json.dump([], f)

if __name__ == '__main__':
    main()

