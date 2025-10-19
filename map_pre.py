import osmnx as ox
def main():
	# bbox order for osmnx v2+: (north, south, east, west)
	bbox = (30.29, 30.26, -97.73, -97.75)
	try:
		# pass bbox as a single tuple (osmnx v2+ expects (north, south, east, west))
		G = ox.graph_from_bbox(bbox, network_type='drive')
		ox.save_graphml(G, "wampus.graphml")
		print('Saved wampus.graphml')
	except Exception as e:
		print('Error creating/saving graph:', e)

if __name__ == '__main__':
	main()