import osmnx as ox
import matplotlib.pyplot as plt

# Load the graph
G = ox.load_graphml("wampus.graphml")

# Extract edge attributes for coloring (e.g. road type)
edge_colors = []
for u, v, data in G.edges(data=True):
    highway = data.get('highway', 'unknown')
    if isinstance(highway, list):
        highway = highway[0]

    # Assign colors by road type
    color_map = {
        'motorway': 'red',
        'primary': 'orange',
        'secondary': 'yellow',
        'tertiary': 'lightgreen', 
        'residential': 'gray',
        'service': 'lightblue',
        'unknown': 'black'
    }
    edge_colors.append(color_map.get(highway, 'black'))

# Plot with color-coded roads
fig, ax = ox.plot_graph(
    G,
    node_size=0,
    edge_color=edge_colors,
    edge_linewidth=1,
    bgcolor='white'
)
plt.show()
