import numpy as np

# Import the updated network structure with acceptance function
exec(open('network-structure-1.py').read())

network = Network()

# Create 10x10 grid with Central Park gap
rows, cols = 10, 10
grid = np.ones((rows, cols), dtype=int)

# Central Park gap (5th to 8th Street, 5th to 8th Avenue)
grid[5:9, 5:9] = 0

# Create adjacency matrix for Manhattan grid
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
                J[current, neighbor] = 100  # Set capacity to 100

network.initialise_junctions(J.shape[0])
network.initialise_roads(J)

# Set road properties with varied densities to create interesting traffic patterns
for idx, road in enumerate(network.roads):
    road.lanes = 1
    road.max_flow = 10  # Set max_flow for acceptance function
    road.potential = 1
    
    # Create varied initial densities (some congested, some free-flowing)
    if idx % 5 == 0:
        road.density = 45  # Heavy congestion (acceptance = 1)
    elif idx % 3 == 0:
        road.density = 30  # Moderate congestion (acceptance = 6)
    elif idx % 2 == 0:
        road.density = 15  # Light traffic (acceptance = 10)
    else:
        road.density = 25  # Light congestion (acceptance = 8)

# Show initial state with acceptance rates
print("Manhattan Traffic Network with Acceptance Function")
print("=" * 60)
print(f"Network has {len(network.roads)} roads and {len(network.junctions)} junctions")

print("\nInitial Traffic Density Distribution:")
density_counts = {}
for road in network.roads:
    density = int(road.density)
    density_counts[density] = density_counts.get(density, 0) + 1

for density, count in sorted(density_counts.items()):
    road_example = next(r for r in network.roads if int(r.density) == density)
    acceptance = road_example.get_acceptance_rate()
    print(f"  {count:3d} roads with density {density}: acceptance rate {acceptance:.1f}")

print(f"\nTotal initial traffic density: {sum(r.density for r in network.roads):.0f}")

print("\nStarting animation with acceptance function...")

# Create and save animated traffic simulation with grid layout
animation = network.save_traffic_animation(
    "manhattan_traffic_with_acceptance.gif", 
    num_steps=20, 
    interval=800,
    grid_layout=True,
    rows=10,
    cols=10
)

# Run manual simulation steps to show acceptance effects
print("\nManual simulation steps showing acceptance function effects:")
Big_A = network.sum_matrices()

for i in range(5):
    print(f'\n{"="*60}')
    print(f'STEP {i+1}')
    print(f'{"="*60}')
    
    if i > 0:
        network.step_forward(Big_A, network.get_current_vector())
    
    # Show overall state
    total_density = sum(road.density for road in network.roads)
    print(f"Total network density: {total_density:.1f}")
    
    # Show congestion analysis
    free_flow = sum(1 for r in network.roads if r.get_acceptance_rate() >= r.max_flow)
    light_cong = sum(1 for r in network.roads if r.max_flow > r.get_acceptance_rate() >= r.max_flow * 0.5)
    heavy_cong = sum(1 for r in network.roads if r.get_acceptance_rate() < r.max_flow * 0.5)
    
    print(f"Road status: {free_flow} free-flowing, {light_cong} light congestion, {heavy_cong} heavy congestion")
    
    # Show total blocked capacity
    total_blocked = sum(r.max_flow - r.get_acceptance_rate() for r in network.roads)
    total_capacity = sum(r.max_flow for r in network.roads)
    blocked_percent = (total_blocked / total_capacity) * 100 if total_capacity > 0 else 0
    
    print(f"Network efficiency: {100-blocked_percent:.1f}% ({blocked_percent:.1f}% capacity blocked by congestion)")
    
    # Show some example roads with different congestion levels
    if i == 0:
        print("\nExample roads (showing acceptance function in action):")
        examples_shown = 0
        for j, road in enumerate(network.roads):
            if examples_shown >= 8:
                break
            acceptance = road.get_acceptance_rate()
            congestion_pct = (1 - acceptance/road.max_flow) * 100
            if j % (len(network.roads) // 8) == 0:  # Show spread of examples
                print(f"  Road {j:2d} (J{road.start}→J{road.end}): density={road.density:4.1f} → acceptance={acceptance:4.1f} ({congestion_pct:4.1f}% blocked)")
                examples_shown += 1

print(f'\n{"="*60}')
print("SIMULATION COMPLETE")
print(f'{"="*60}')
print("\nThe acceptance function creates realistic traffic behavior:")
print("✓ Congested roads (density > 20) limit incoming traffic")
print("✓ This causes upstream backup when roads can't accept more cars")
print("✓ Traffic flow becomes self-regulating based on congestion levels")
print("✓ Animation shows these effects visually with color intensity")
