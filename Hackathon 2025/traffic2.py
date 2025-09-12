import numpy as np
import networkstructure1 as ns

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
    road.max_flow =  (idx+1)*10 # Set a default max_flow for all roads

junctions = network.junctions

for road in network.roads:
    road.density = 20
    road.potential = 10


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

print("\nInitial Acceptance Rates (showing first 10 roads):")
for i in range(min(10, len(network.roads))):
    road = network.roads[i]
    acceptance = road.get_acceptance_rate()
    print(f"  Road {i} (J{road.start}→J{road.end}): density={road.density:.1f} → acceptance={acceptance:.1f}")

# Calculate initial total density for mass conservation check
initial_total_density = sum(road.density for road in network.roads)
print(f"\nInitial total density: {initial_total_density:.6f}")

print("\nStarting manual simulation with mass conservation checks...")

# Run simulation steps with mass conservation checking
Big_A = network.sum_matrices()
for i in range(5):
    print(f'\nSTEP {i+1}:')
    
    # Check mass before step
    before_total = sum(road.density for road in network.roads)
    
    # Take a step
    network.step_forward(Big_A, network.get_current_vector())
    
    # Check mass after step
    after_total = sum(road.density for road in network.roads)
    mass_difference = abs(after_total - before_total)
    
    print(f"  Mass before: {before_total:.6f}")
    print(f"  Mass after:  {after_total:.6f}")
    print(f"  Difference:  {mass_difference:.2e}")
    
    if mass_difference > 1e-10:
        print(f"  ⚠️  WARNING: Mass not conserved!")
    else:
        print(f"  ✅ Mass conserved")
    
    network.show_current_state(compact=True)
    
    # Show how acceptance rates change
    print("  Acceptance rates (first 5 roads):")
    for j in range(min(5, len(network.roads))):
        road = network.roads[j]
        acceptance = road.get_acceptance_rate()
        congestion = (1 - acceptance/road.max_flow) * 100 if road.max_flow > 0 else 0
        print(f"    Road {j}: {acceptance:.1f} ({congestion:.0f}% congestion)")

print("\nFinal acceptance analysis:")
network.show_acceptance_rates()

# Final mass conservation check
final_total_density = sum(road.density for road in network.roads)
total_difference = abs(final_total_density - initial_total_density)
print(f"\nFINAL MASS CONSERVATION CHECK:")
print(f"Initial total density: {initial_total_density:.6f}")
print(f"Final total density:   {final_total_density:.6f}")
print(f"Total difference:      {total_difference:.2e}")

if total_difference > 1e-10:
    print("❌ OVERALL MASS NOT CONSERVED!")
else:
    print("✅ OVERALL MASS CONSERVED!")
