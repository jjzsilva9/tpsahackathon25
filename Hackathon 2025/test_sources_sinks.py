import numpy as np
import network_structure_2 as ns

print("Testing Sources and Sinks Functionality")
print("=" * 50)

# Create a simple test network
network = ns.Network()

# Create 10x10 grid with Central Park gap
rows, cols = 10, 10
grid = np.ones((rows, cols), dtype=int)

# Central Park gap (5th to 8th Street, 5th to 8th Avenue)
grid[5:9, 5:9] = 0

# Create adjacency matrix
total_nodes = rows * cols
J = np.zeros((total_nodes, total_nodes), dtype=int)

def get_index(i, j):
    return i * cols + j

for i in range(rows):
    for j in range(cols):
        if grid[i, j] == 0:  # Skip Central Park nodes
            continue
        
        current = get_index(i, j)
        
        # Check 4 nearest neighbors
        neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
        
        for ni, nj in neighbors:
            if 0 <= ni < rows and 0 <= nj < cols and grid[ni, nj] == 1:
                neighbor = get_index(ni, nj)
                J[current, neighbor] = 1

network.initialise_junctions(J.shape[0])
network.initialise_roads(J)

# Set initial densities and potentials
center_junction = 44
center_row = center_junction // cols
center_col = center_junction % cols

for road in network.roads:
    road.density = 50
    road.max_flow = 10
    
    # Calculate distance and set potential
    start_row = road.start // cols
    start_col = road.start % cols
    end_row = road.end // cols
    end_col = road.end % cols
    
    start_distance = abs(start_row - center_row) + abs(start_col - center_col)
    end_distance = abs(end_row - center_row) + abs(end_col - center_col)
    avg_distance = (start_distance + end_distance) / 2
    road.potential = 10 + (avg_distance * 10)

# Define source and sink roads
road_ids_sink = ['J44J45','J44J54','J45J46','J45J35','J44J34','J44J43','J45J44','J54J44']
road_ids_source = ['J0J1','J0J10','J90J80','J90J91','J9J8','J9J19','J99J89','J99J98']

print(f"Initial total density: {sum(road.density for road in network.roads):.1f}")
print()

# Calculate A matrix
A_total = network.sum_matrices()

# Manually test 3 simulation steps
for step in range(3):
    print(f"STEP {step + 1}:")
    
    # Record densities before
    sink_densities_before = []
    source_densities_before = []
    
    for road_id in road_ids_sink:
        road = network.roads_dict.get(road_id)
        if road:
            sink_densities_before.append(road.density)
    
    for road_id in road_ids_source:
        road = network.roads_dict.get(road_id)
        if road:
            source_densities_before.append(road.density)
    
    total_before = sum(road.density for road in network.roads)
    
    # Step simulation
    current_vector = network.get_current_vector()
    network.step_forward(A_total, current_vector)
    
    # Apply sources and sinks
    for road_id_sink in road_ids_sink:
        road = network.roads_dict.get(road_id_sink)
        if road:
            road.density = max(0, road.density - 10)  # Remove cars
    
    for road_id_source in road_ids_source:
        road = network.roads_dict.get(road_id_source)
        if road:
            road.density += 10  # Add cars
    
    # Record densities after
    sink_densities_after = []
    source_densities_after = []
    
    for road_id in road_ids_sink:
        road = network.roads_dict.get(road_id)
        if road:
            sink_densities_after.append(road.density)
    
    for road_id in road_ids_source:
        road = network.roads_dict.get(road_id)
        if road:
            source_densities_after.append(road.density)
    
    total_after = sum(road.density for road in network.roads)
    
    # Show changes
    print(f"  Total density before: {total_before:.1f}")
    print(f"  Total density after:  {total_after:.1f}")
    print(f"  Net change: {total_after - total_before:.1f}")
    
    print(f"  SINK roads (should decrease by ~10 each):")
    for i, road_id in enumerate(road_ids_sink):
        if i < len(sink_densities_before) and i < len(sink_densities_after):
            change = sink_densities_after[i] - sink_densities_before[i]
            print(f"    {road_id}: {sink_densities_before[i]:.1f} → {sink_densities_after[i]:.1f} (change: {change:.1f})")
    
    print(f"  SOURCE roads (should increase by 10 each):")
    for i, road_id in enumerate(road_ids_source):
        if i < len(source_densities_before) and i < len(source_densities_after):
            change = source_densities_after[i] - source_densities_before[i]
            print(f"    {road_id}: {source_densities_before[i]:.1f} → {source_densities_after[i]:.1f} (change: {change:.1f})")
    
    print()

print("Test completed!")
