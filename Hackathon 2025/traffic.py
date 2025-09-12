import numpy as np
import network_structure as ns

network = ns.Network()

# Create 10x10 grid with Central Park gap
rows, cols = 10, 10
grid = np.ones((rows, cols), dtype=int)

# Central Park gap (5th to 8th Street, 5th to 8th Avenue)
grid[5:9, 5:9] = 0

# Create adjacency matrix for Ising model
# Only nearest neighbors (up, down, left, right) connected
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
for idx, road in enumerate(network.roads):
    road.lanes = 1
    road.max_flow =  10 # Set a default max_flow for all roads

junctions = network.junctions

# Set densities and calculate radial potentials from junction 44
center_junction = 44
center_row = center_junction // cols  # Row of junction 44
center_col = center_junction % cols   # Column of junction 44

for road in network.roads:
    road.density = 50
    
    # Calculate distance from junction 44 for both start and end junctions
    start_row = road.start // cols
    start_col = road.start % cols
    end_row = road.end // cols
    end_col = road.end % cols
    
    # Calculate Manhattan distance from center (junction 44) for both endpoints
    start_distance = abs(start_row - center_row) + abs(start_col - center_col)
    end_distance = abs(end_row - center_row) + abs(end_col - center_col)
    
    # Use the average distance of the road's endpoints
    avg_distance = (start_distance + end_distance) / 2

    # Set potential: starting at 10 for distance 0, increasing by 100 for each unit distance
    road.potential = 10 + (avg_distance * 100)

A_1_default = network.calculate_A(junctions[1].id)
A_2_default = network.calculate_A(junctions[2].id)
A_3_default = network.calculate_A(junctions[3].id)
A_4_default = network.calculate_A(junctions[0].id) 

A_1_random = network.calculate_A(junctions[1].id)
A_2_random = network.calculate_A(junctions[2].id)
A_3_random = network.calculate_A(junctions[3].id)
A_4_random = network.calculate_A(junctions[0].id) 

# Show initial state
print("Initial Network State:")
network.show_current_state(compact=True)
print("\nStarting animation...")

# Create and save animated traffic simulation with grid layout
animation = network.save_traffic_animation(
    "manhattan_traffic.gif", 
    num_steps=15, 
    interval=1000,
    grid_layout=True,
    rows=10,
    cols=10
)

# Also run a few steps manually to show the progression
print("\nManual simulation steps:")
Big_A = network.sum_matrices()
for i in range(3):
    print(f'\nSTEP {i+1}:')
    network.step_forward(Big_A, network.get_current_vector())
    network.show_current_state(compact=True)

# Create and save animated traffic simulation with improved settings
print("\nCreating animation with improved visual settings...")
animation = network.save_traffic_animation(
    "random_network_traffic.gif", 
    num_steps=30, 
    interval=1000,
    grid_layout=True,  # Use spring layout for random network
    rows=10,
    cols=10
)
