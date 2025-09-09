import osmnx as ox
import networkx as nx
import folium
import numpy as np
import pandas as pd
import math

# Download and prepare the graph
place_name = "Dublin City, Ireland"
G = ox.graph.graph_from_place(place_name, network_type="drive")

# G = ox.simplification.consolidate_intersections(
#        G,
#        tolerance=0.0002,
#        rebuild_graph=True,
#        dead_ends=False 
#    )

# Convert graph to GeoDataFrames
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

print(f'Node info: {nodes_gdf.info()}')
print(f'Edge info: {edges_gdf.info()}')

A = nx.to_numpy_array(G)
np.save("adj_matrix", A)


# Calculate map center
center_lat = (nodes_gdf.y.min() + nodes_gdf.y.max()) / 2
center_lon = (nodes_gdf.x.min() + nodes_gdf.x.max()) / 2

# Create the map
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=12,
    tiles='OpenStreetMap'
)

# Add edges to the map
for idx, row in edges_gdf.iterrows():
    if row.geometry is not None:

        # Convert coordinates for Folium [lat, lon] format
        coords = [[point[1], point[0]] for point in row.geometry.coords]
        
        # Create popup with street information
        popup_text = f"Street: {row.get('name', 'Unnamed')}<br>Type: {row.get('highway', 'Unknown')}<br>Reversed: {row.get('reversed', 'Unknown')}"


        if idx[0] < 20:
            print(f'Num lanes : {row.lanes}')
            print(f'Type : {row.highway}')
            try:
                folium.PolyLine(
            locations=coords,
            color='orange',
            weight=2,
            opacity=0.7,
            popup=folium.Popup(popup_text, max_width=200)
        ).add_to(m)
            except:        
                folium.PolyLine(
                    locations=coords,
                    color='blue',
                    weight=2,
                    opacity=0.7,
                    popup=folium.Popup(popup_text, max_width=200)
                ).add_to(m)

        else:
            folium.PolyLine(
            locations=coords,
            color='blue',
            weight=2,
            opacity=0.7,
            popup=folium.Popup(popup_text, max_width=200)
        ).add_to(m)

# Add some major intersections as markers
major_intersections = [node for node, degree in G.degree() if degree >= 8]
for node_id in major_intersections:
    node_data = nodes_gdf.loc[node_id]
    folium.CircleMarker(
        location=[node_data.y, node_data.x],
        radius=4,
        popup=f"Major Intersection<br>Node: {node_id}<br>Connections: {G.degree(node_id)}",
        color='red',
        fillColor='red',
        fillOpacity=0.8
    ).add_to(m)

# Visualize sensor locations from dlr_scats_sites-1.csv
sensor_df = pd.read_csv("scats-data/dcc_traffic_signals_20221130.csv")
    
for idx, row in sensor_df.iterrows():
    lat, lon = row.get('Lat'), row.get('Long')
    site_id, site_type = row.get('Site_ID'), row.get("Site_Type")
    if lat is not None and lon is not None and "SCATS" in site_type:
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            popup=f"Sensor ID: {site_id}",
            colour='green',
            fillColor='green',
            fillOpacity=0.8
        ).add_to(m)

nodes_gdf.to_csv("node_data.csv")
edges_gdf.to_csv("edges_data.csv")

# Save the map
m.save("dublin_interactive_osmnx.html")
print("Interactive map saved as 'dublin_interactive_osmnx.html'")

# Map each SCATS site to the nearest node using Euclidean distance
scats_df = pd.read_csv("scats-data/dcc_traffic_signals_20221130.csv")
nodes_df = pd.read_csv("node_data.csv")

mapping = []
for idx, scats_row in scats_df.iterrows():
    site_id = scats_row['SiteID']
    site_lat = scats_row['Lat']
    site_lon = scats_row['Long']
    # Compute Euclidean distance to all nodes
    dists = ((nodes_df['y'] - site_lat)**2 + (nodes_df['x'] - site_lon)**2)
    min_idx = dists.idxmin()
    nearest_node = nodes_df.loc[min_idx]
    mapping.append({
        'SiteID': site_id,
        'Node_osmid': nearest_node['osmid'],
        'Site_Lat': site_lat,
        'Site_Lon': site_lon,
        'Node_Lat': nearest_node['y'],
        'Node_Lon': nearest_node['x'],
        'Distance': math.sqrt(dists[min_idx])
    })

mapping_df = pd.DataFrame(mapping)
mapping_df.to_csv("scats_to_node_mapping.csv", index=False)
print("SCATS site to node mapping saved as 'scats_to_node_mapping.csv'")
# Display basic graph info
print(f"Graph contains {len(G.nodes)} nodes and {len(G.edges)} edges")
# print(f"Found {len(major_intersections)} major intersections (degree >= 4)")
