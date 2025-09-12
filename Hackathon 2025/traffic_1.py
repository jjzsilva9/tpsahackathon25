import numpy as np
import network_structure as ns

network = ns.Network()

def create_adjacency_matrix_random(num_nodes, connection_prob=0.8, min_weight=1, max_weight=10):
    """Creates a random weighted adjacency matrix for an undirected (bidirectional) graph."""
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < connection_prob:
                weight = np.random.randint(min_weight, max_weight + 1)
                adj_matrix[i, j] = weight
                adj_matrix[j, i] = weight
    return adj_matrix

def create_adjacency_matrix(num_nodes, connection_prob=0.3):
    """Creates a random adjacency matrix for an undirected (bidirectional) graph."""
    adj_matrix = np.random.rand(num_nodes, num_nodes) < connection_prob
    np.fill_diagonal(adj_matrix, 0) 
    adj_matrix = np.triu(adj_matrix, 1)
    adj_matrix = (adj_matrix + adj_matrix.T)
    return adj_matrix.astype(int)

J = create_adjacency_matrix_random(5, connection_prob=0.7)
print("Adjacency Matrix:")
print(J)
network.initialise_junctions(J.shape[0])

network.initialise_roads(J)
for road in network.roads:
    road.lanes = 1
    road.max_flow = 10  # Set a default max_flow for all roads

junctions = network.junctions

# Method 5: Random assignment within ranges
def randomize_outgoing_roads(network, junction_id, density_range=(0, 5), potential_range=(0.5, 2.0)):
    """Randomly assign density and potential values to outgoing roads"""
    junction = network.junctions[junction_id]
    print(f"\nRandomizing outgoing roads from Junction {junction_id}:")
    
    for road_id in junction.roads_out:
        road = network.get_road_by_id(road_id)
        if road:
            road.density = np.random.uniform(*density_range)
            road.potential = np.random.uniform(*potential_range)
            print(f"  Road {road_id}: density={road.density:.2f}, potential={road.potential:.2f}")

# First, let's see the A matrices with default values (density=0, potential=1)
print("\n" + "="*60)
print("A MATRICES WITH DEFAULT VALUES")
print("="*60)

A_1_default = network.calculate_A(junctions[1].id)
A_2_default = network.calculate_A(junctions[2].id)
A_3_default = network.calculate_A(junctions[3].id)
A_4_default = network.calculate_A(junctions[0].id) 
print("Matrix A for junction 1 (default):")
print(A_1_default)
print("Matrix A for junction 2 (default):")
print(A_2_default)
print("Matrix A for junction 3 (default):")
print(A_3_default)

# Now randomize the road properties
print("\n" + "="*60)
print("RANDOMIZING ROAD PROPERTIES")
print("="*60)

# Randomize roads from all junctions to see the effect
for junction_id in range(len(junctions)):
    randomize_outgoing_roads(network, junction_id, density_range=(1, 5), potential_range=(0.5, 2.0))

# Now calculate A matrices with the new randomized values
print("\n" + "="*60)
print("A MATRICES WITH RANDOMIZED VALUES")
print("="*60)

A_1_random = network.calculate_A(junctions[1].id)
A_2_random = network.calculate_A(junctions[2].id)
A_3_random = network.calculate_A(junctions[3].id)
A_4_random = network.calculate_A(junctions[0].id) 
print("Matrix A for junction 1 (randomized):")
print(A_1_random)
print("Matrix A for junction 2 (randomized):")
print(A_2_random)
print("Matrix A for junction 3 (randomized):")
print(A_3_random)


for i in range(5):
    print(f'THIS IS STEP {i+1}')
    Big_A = network.sum_matrices()
    network.step_forward(Big_A, network.get_current_vector())
    network.show_current_state(compact=True)

# Create and save animated traffic simulation with improved settings
print("\nCreating animation with improved visual settings...")
animation = network.save_traffic_animation(
    "random_network_traffic.gif", 
    num_steps=10, 
    interval=1000,
    grid_layout=False,  # Use spring layout for random network
    rows=5,
    cols=5
)

