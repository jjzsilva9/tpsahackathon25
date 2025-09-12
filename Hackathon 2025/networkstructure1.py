import numpy as np

class Road:
    # 
    def __init__(self, start, end, id, capacity, potential, name=None, density=0, details=None, max_flow=10, lanes=1):
        self.start = start
        self.end = end
        self.id = id
        self.name = name
        self.density = density  # Use the parameter value instead of hardcoding to 0
        self.flow_capacity = capacity
        self.potential = potential
        self.details = details
        self.max_flow = max_flow
        self.lanes = lanes
        return

    #Returns True (False) if full (not full)
    def is_full(self):
        return self.density >= self.flow_capacity
    
    def update_occ(self,new_occ):
        self.density = new_occ
        return
    
    def get_acceptance_rate(self):
        """
        Calculate acceptance rate based on current density.
        - For density < 20: acceptance = max_flow (no congestion)
        - For density 20-42.5: linear drop from max_flow to 1
        - For density > 42.5: acceptance = 1 (heavy congestion)
        """
        if self.density < 20:
            return self.max_flow
        elif self.density <= 42.5:
            # Linear interpolation from max_flow at density=20 to 1 at density=42.5
            slope = (1 - self.max_flow) / (42.5 - 20)
            acceptance = self.max_flow + slope * (self.density - 20)
            return max(1, acceptance)  # Ensure minimum of 1
        else:
            return 1
    
    def __str__(self):
        return f"Road({self.start}->{self.end}, cars={self.density}/{self.flow_capacity})"
        
class Junction:
    def __init__(self, id, name=None):
        self.id = id
        self.name = name
        self.roads_out = []  # Changed to set for easier road management
        self.road_in = []
        return

    def __repr__(self):
        return f"Junction({self.id})"

class Network:
    def __init__(self):
        self.junctions = []
        self.roads = []  # Ordered list for vector operations
        self.roads_dict = {}  # Dictionary for fast lookup by road_id
        self.state_vector = []
        pass

    def add_junction(self, id, name = None):
        self.junctions.append(Junction(id,name=name))
        return

    def initialise_junctions(self, num_junctions):
        for i in range(num_junctions):
            self.junctions.append(Junction(id=i))

    def add_road(self, start_id, end_id, capacity, potential=1, density=0, name=None):
        road_id = f'J{start_id}J{end_id}'
        road = Road(start_id, end_id, road_id, capacity, potential, name=name, density=density)
        
        # Add to both structures for dual access
        self.roads.append(road)  # Maintain order for vectors
        self.roads_dict[road_id] = road  # Fast lookup by ID
        self.state_vector.append(road.density)
        
        # Connect the road to both junctions
        self.junctions[start_id].roads_out.append(road_id)
        self.junctions[end_id].road_in.append(road_id)

    def get_current_vector(self):
        """Get current values for all roads as a numpy array"""
        current_vector = np.array([min(road.density, road.max_flow) for road in self.roads])
        return current_vector

    def get_road_by_id(self, road_id):
        """Get road object by road_id efficiently using dictionary lookup"""
        return self.roads_dict.get(road_id)
    
    def get_road_index(self, road_id):
        """Get the index of a road in the ordered list by road_id"""
        road = self.roads_dict.get(road_id)
        if road:
            return self.roads.index(road)
        return None
    
    def get_road_by_index(self, index):
        """Get road object by index in the ordered list"""
        if 0 <= index < len(self.roads):
            return self.roads[index]
        return None


    def initialise_roads(self, adj_matrix):
        """
        Initialize roads based on an adjacency matrix.
        
        Args:
            adj_matrix: 2D array/list where element (i,j) represents the capacity
                       of the road from junction i to junction j.
                       Capacity = 0 means no road exists between junctions.
        """
        # Convert to numpy array for easier handling
        matrix = np.array(adj_matrix)
        num_junctions = matrix.shape[0]
        
        # Ensure we have the required number of junctions
        if len(self.junctions) < num_junctions:
            raise ValueError('Not enough junctions for adjacency matrix size.')
        
        # Initialize roads based on the adjacency matrix
        road_count = 0
        for i in range(num_junctions):
            for j in range(num_junctions):
                capacity = matrix[i][j]
                
                # Only create road if capacity > 0 (non-zero capacity means road exists)
                if capacity > 0:
                    # Create a human-readable road name
                    road_name = f"J{i}J{j}"
                    
                    # Add the road to the network
                    self.add_road(
                        start_id=i, 
                        end_id=j, 
                        capacity=capacity, 
                        density=0, 
                        name=road_name
                    )
                    road_count += 1
        
        print(f"Initialized {road_count} roads from adjacency matrix")
        return road_count

    def visualize_network(self, show_details=True):
        """
        Display a nice visualization of the network showing junctions and roads.
        
        Args:
            show_details (bool): Whether to show detailed road information
        """
        print("=" * 60)
        print("NETWORK VISUALIZATION")
        print("=" * 60)
        
        if not self.junctions:
            print("No junctions in the network.")
            return
        
        # Display junction summary
        print(f"📍 JUNCTIONS ({len(self.junctions)} total)")
        print("-" * 30)
        for junction in self.junctions:
            junction_name = junction.name if junction.name else f"Junction_{junction.id}"
            road_count = len(junction.roads_out) + len(junction.road_in)
            print(f"  J{junction.id}: {junction_name} ({road_count} roads)")
        
        print()
        
        # Display road summary
        print(f"🛣️  ROADS ({len(self.roads)} total)")
        print("-" * 50)
        
        if not self.roads:
            print("  No roads in the network.")
        else:
            for road in self.roads:
                # Calculate density percentage
                occ_percent = (road.density / road.flow_capacity * 100) if road.flow_capacity > 0 else 0
                
                # Create visual indicator for road load
                if occ_percent == 0:
                    status = "🟢 Empty"
                elif occ_percent < 50:
                    status = "🟡 Light"
                elif occ_percent < 90:
                    status = "🟠 Busy"
                else:
                    status = "🔴 Full"
                
                # Road direction arrow
                arrow = "──>"
                
                road_display = f"  J{road.start} {arrow} J{road.end}"
                
                if show_details:
                    capacity_info = f"[{road.density}/{road.flow_capacity}] {status}"
                    print(f"{road_display:<15} {capacity_info}")
                else:
                    print(f"{road_display}")
        
        print()
        
        # Display adjacency matrix representation
        if self.junctions:
            print("📊 ADJACENCY MATRIX (Capacities)")
            print("-" * 40)
            
            # Create adjacency matrix for visualization
            max_junc_id = max(j.id for j in self.junctions)
            adj_matrix = np.zeros((max_junc_id + 1, max_junc_id + 1), dtype=int)
            
            # Fill the matrix with road capacities
            for road in self.roads:
                adj_matrix[road.start][road.end] = road.flow_capacity
            
            # Print header row
            print("    ", end="")
            for j in range(max_junc_id + 1):
                print(f"J{j:2d}", end=" ")
            print()
            
            # Print matrix rows
            for i in range(max_junc_id + 1):
                print(f"J{i:2d}:", end=" ")
                for j in range(max_junc_id + 1):
                    capacity = adj_matrix[i][j]
                    if capacity == 0:
                        print("  -", end=" ")
                    else:
                        print(f"{capacity:3d}", end=" ")
                print()
        
        print("=" * 60)

    def show_current_state(self, compact=True):
        """
        Display current network state showing road densities in a simple format.
        
        Args:
            compact (bool): If True, show simple list format. If False, show detailed format.
        """
        if compact:
            print("Current Network State:")
            
            # Calculate total density
            total_density = sum(road.density for road in self.roads)
            
            # Show roads in simple format
            for i, road in enumerate(self.roads):
                density_str = f"{road.density:.2f}" if road.density != int(road.density) else f"{int(road.density)}"
                print(f"road {i} (J{road.start} --> J{road.end}): {density_str}")
            
            print(f"total_density = {total_density:.2f}")
        
        else:
            # Detailed format
            print("DETAILED NETWORK STATE")
            print("-" * 60)
            
            total_density = 0
            total_capacity = 0
            
            for road in self.roads:
                density = road.density
                capacity = road.flow_capacity
                acceptance = road.get_acceptance_rate()
                total_density += density
                total_capacity += capacity
                
                fill_percent = (density / capacity * 100) if capacity > 0 else 0
                
                # Status description based on acceptance rate
                if acceptance >= road.max_flow:
                    status = "Free Flow"
                elif acceptance > road.max_flow * 0.5:
                    status = "Light Congestion"
                elif acceptance > 2:
                    status = "Moderate Congestion" 
                else:
                    status = "Heavy Congestion"
                
                print(f"  Road J{road.start}→J{road.end}:")
                print(f"    Density: {density:.2f}/{capacity} ({fill_percent:.1f}%)")
                print(f"    Acceptance Rate: {acceptance:.2f}/{road.max_flow}")
                print(f"    Status: {status}")
                print(f"    Potential: {road.potential:.2f}")
                print()
            
            overall_percent = (total_density / total_capacity * 100) if total_capacity > 0 else 0
            print(f"NETWORK SUMMARY:")
            print(f"  Total Density: {total_density:.2f}")
            print(f"  Total Capacity: {total_capacity}")
            print(f"  Overall Fill: {overall_percent:.1f}%")
            print("-" * 60)
    
    def show_acceptance_rates(self):
        """
        Display acceptance rates for all roads to visualize traffic congestion.
        """
        print("ROAD ACCEPTANCE RATES (Traffic Congestion Analysis)")
        print("-" * 70)
        
        for i, road in enumerate(self.roads):
            acceptance = road.get_acceptance_rate()
            density = road.density
            max_flow = road.max_flow
            
            # Calculate congestion level
            congestion_percent = (1 - acceptance/max_flow) * 100 if max_flow > 0 else 0
            
            # Visual indicator
            if congestion_percent == 0:
                indicator = "🟢 Free"
            elif congestion_percent < 25:
                indicator = "🟡 Light"
            elif congestion_percent < 50:
                indicator = "🟠 Moderate"
            else:
                indicator = "🔴 Heavy"
            
            print(f"Road {i:2d} (J{road.start}→J{road.end}): "
                  f"Density={density:5.1f} → Acceptance={acceptance:4.1f}/{max_flow} "
                  f"({congestion_percent:4.1f}% congestion) {indicator}")
        
        print("-" * 70)

    def calculate_A(self, junc_id):
        working_roads_in = self.junctions[junc_id].road_in  # Fixed: road_in not roads_in
        working_roads_out = self.junctions[junc_id].roads_out
        all_working_roads = working_roads_in + working_roads_out
        num_roads_total = len(all_working_roads)
        
        A = np.zeros((num_roads_total,num_roads_total))

        denom = 0
        for out_road_id in working_roads_out:
            denom += 1/(self.roads_dict[out_road_id].density + self.roads_dict[out_road_id].potential)  # Fixed: consistent dict access
        
        # Fixed: moved matrix element calculation inside the loops where variables are defined
        for idx_in, road_id in enumerate(all_working_roads):
            if road_id in working_roads_in:
                mat_element = -1
                A[idx_in,idx_in] += mat_element
                for idx_out, road_id_out in enumerate(all_working_roads):
                    if road_id_out in working_roads_out:  # Fixed: check road_id_out not road_id
                        mat_element = (1/(self.roads_dict[road_id_out].density + self.roads_dict[road_id_out].potential))/(denom)  # Fixed: consistent dict access and correct variable
                        A[idx_out,idx_in] += mat_element

        return A
    
    def embed_A(self, junc_id):
        """
        Calculate junction matrix A embedded in a global matrix.
        
        Args:
            junc_id: Junction ID to calculate matrix for
            
        Returns:
            A matrix of size (total_roads x total_roads) where only elements
            corresponding to roads connected to this junction are non-zero
        """
        # Get the local junction matrix using the existing function
        local_A = self.calculate_A(junc_id)
        
        # Get roads connected to this junction
        working_roads_in = self.junctions[junc_id].road_in  # Fixed: road_in not roads_in
        working_roads_out = self.junctions[junc_id].roads_out
        all_working_roads = working_roads_in + working_roads_out
        
        # Create global matrix (same size as total number of roads)
        total_roads = len(self.roads)
        global_A = np.zeros((total_roads, total_roads))
        
        # Get global indices for the working roads
        global_indices = []
        for road_id in all_working_roads:
            global_index = self.get_road_index(road_id)
            if global_index is not None:
                global_indices.append(global_index)
        
        # Embed the local matrix into the global matrix
        for i, global_i in enumerate(global_indices):
            for j, global_j in enumerate(global_indices):
                global_A[global_i, global_j] = local_A[i, j]
        
        return global_A

    def sum_matrices(self):
        total_A = sum(self.embed_A(junc_id) for junc_id in range(len(self.junctions)))
        return total_A

    def step_forward(self, A_total, current_vector):
        """
        Step forward with acceptance function to model traffic congestion.
        Cars can only move if the destination road can accept them.
        Mass conservation is strictly maintained with numerical precision handling.
        """
        # Store initial total mass for verification
        initial_total_mass = sum(road.density for road in self.roads)
        
        # Calculate desired flow changes
        v_delta = A_total @ current_vector
        
        # Apply acceptance function - process each road's incoming flow
        actual_delta = np.zeros_like(v_delta, dtype=np.float64)  # Use double precision
        
        # Track total rejected cars for mass conservation check
        total_rejected = 0.0
        
        # For each road, check if it can accept the incoming cars
        for i, road in enumerate(self.roads):
            desired_incoming = v_delta[i]
            
            if desired_incoming > 0:  # Cars want to enter this road
                # Check acceptance rate
                acceptance_rate = road.get_acceptance_rate()
                actual_incoming = min(desired_incoming, acceptance_rate)
                actual_delta[i] = actual_incoming
                
                # The difference (rejected cars) must be redistributed back to source roads
                rejected_cars = desired_incoming - actual_incoming
                total_rejected += rejected_cars
                
                # Find which roads were trying to send cars to this road and redistribute properly
                if rejected_cars > 1e-12:  # Only process if rejection is significant
                    # Calculate total flow that wanted to come to road i
                    total_incoming_flow = 0.0
                    source_flows = {}
                    
                    for j in range(len(self.roads)):
                        if A_total[i, j] > 1e-12:  # Only consider significant flows
                            flow_from_j = A_total[i, j] * current_vector[j]
                            if flow_from_j > 1e-12:
                                source_flows[j] = flow_from_j
                                total_incoming_flow += flow_from_j
                    
                    # Redistribute rejected cars proportionally back to source roads
                    if total_incoming_flow > 1e-12:
                        redistributed_total = 0.0
                        for j, flow_from_j in source_flows.items():
                            proportion = flow_from_j / total_incoming_flow
                            rejected_for_this_source = rejected_cars * proportion
                            actual_delta[j] += rejected_for_this_source
                            redistributed_total += rejected_for_this_source
                        
                        # Handle any remaining mass due to floating point precision
                        mass_error = rejected_cars - redistributed_total
                        if abs(mass_error) > 1e-12 and source_flows:
                            # Add the error to the largest source flow proportionally
                            largest_source = max(source_flows, key=source_flows.get)
                            actual_delta[largest_source] += mass_error
            else:
                # Cars leaving this road (negative flow) - always allowed
                actual_delta[i] = desired_incoming
        
        # Update road densities with actual (accepted) flows
        for i, road in enumerate(self.roads):
            new_density = road.density + actual_delta[i]
            road.update_occ(max(0.0, new_density))  # Ensure non-negative density
        
        # Verify mass conservation
        final_total_mass = sum(road.density for road in self.roads)
        mass_difference = abs(final_total_mass - initial_total_mass)
        
        if mass_difference > 1e-10:
            print(f"WARNING: Mass conservation error: {mass_difference:.2e}")
            print(f"  Initial mass: {initial_total_mass:.12f}")
            print(f"  Final mass: {final_total_mass:.12f}")
            print(f"  Total rejected cars: {total_rejected:.12f}")
        
        # Return the new density vector
        new_densities = np.array([road.density for road in self.roads])
        return new_densities

    def animate_traffic_flow(self, num_steps=30, interval=800, figsize=(10, 8)):
        """
        Create a simple animated visualization showing road densities over time.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            import networkx as nx
            
            # Set backend for better compatibility
            import matplotlib
            matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on system
            
        except ImportError:
            print("Error: matplotlib and networkx required. Install with: pip install matplotlib networkx")
            return
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes (junctions with roads)
        for junction in self.junctions:
            if junction.road_in or junction.roads_out:
                G.add_node(junction.id)
        
        # Add edges (roads)
        edge_road_mapping = {}
        for road in self.roads:
            G.add_edge(road.start, road.end)
            edge_road_mapping[(road.start, road.end)] = road
        
        # Setup plot
        fig, ax = plt.subplots(figsize=figsize)
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Calculate A matrix for simulation
        A_total = self.sum_matrices()
        initial_densities = [road.density for road in self.roads]
        
        def update_frame(frame_num):
            ax.clear()
            
            # Step simulation
            if frame_num > 0:
                current_vector = self.get_current_vector()
                self.step_forward(A_total, current_vector)
            
            # Get road densities for edge labels and colors
            edge_labels = {}
            edge_colors = []
            for edge in G.edges():
                road = edge_road_mapping[edge]
                edge_labels[edge] = f"{road.density:.1f}"
                
                # Color roads based on density (normalized by max density)
                max_density = max(r.density for r in self.roads)
                if max_density > 0:
                    color_intensity = road.density / max_density
                    edge_colors.append(plt.cm.Reds(color_intensity))
                else:
                    edge_colors.append('gray')
            
            # Draw network
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=600)
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=2, 
                                 arrowsize=15, arrowstyle='->')
            nx.draw_networkx_labels(G, pos, {node: f"J{node}" for node in G.nodes()}, 
                                  ax=ax, font_size=8)
            
            # Only show edge labels for a subset to avoid clutter
            if len(edge_labels) < 50:  # Only show labels if network is small enough
                nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=6, 
                                           bbox=dict(boxstyle='round,pad=0.1', facecolor='yellow', alpha=0.7))
            
            ax.set_title(f"Traffic Densities - Step {frame_num}\n(Red intensity = traffic density)", fontsize=14)
            ax.set_aspect('equal')
            
            # Add total density info
            total_density = sum(road.density for road in self.roads)
            ax.text(0.02, 0.98, f"Total Density: {total_density:.2f}", transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10, verticalalignment='top')
        
        # Create and show animation
        ani = animation.FuncAnimation(fig, update_frame, frames=num_steps, 
                                    interval=interval, repeat=True)
        
        # Reset to initial state
        for i, density in enumerate(initial_densities):
            self.roads[i].density = density
        
        plt.tight_layout()
        plt.show()
        return ani

    def save_traffic_animation(self, filename="traffic_animation.gif", num_steps=20, interval=800, grid_layout=True, rows=10, cols=10):
        """
        Create and save traffic flow animation as a GIF file.
        
        Args:
            filename: Base filename for the animation
            num_steps: Number of animation steps
            interval: Time between frames in ms
            grid_layout: Whether to use grid layout for nodes
            rows: Number of rows in the grid (for grid_layout=True)
            cols: Number of columns in the grid (for grid_layout=True)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            import networkx as nx
            from datetime import datetime
            
            # Set backend for file saving
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend for saving files
            
        except ImportError:
            print("Error: matplotlib and networkx required.")
            return
        
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, 'gif')
        timestamped_filename = f"{base_name}_{timestamp}.{ext}"
        
        print(f"Creating traffic flow animation with timestamp: {timestamped_filename}")
        
        # Create a graph for the network
        G = nx.DiGraph()
        
        # Add all nodes that have roads connected
        active_junctions = set()
        for road in self.roads:
            active_junctions.add(road.start)
            active_junctions.add(road.end)
        
        for junction_id in active_junctions:
            G.add_node(junction_id)
        
        # Add edges (roads)
        edge_road_mapping = {}
        for road in self.roads:
            if road.start in active_junctions and road.end in active_junctions:
                G.add_edge(road.start, road.end)
                edge_road_mapping[(road.start, road.end)] = road
        
        # Setup plot with grid layout
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        if grid_layout:
            # Create grid positions for nodes (Manhattan-like layout)
            pos = {}
            max_node = max(active_junctions) if active_junctions else 99
            
            # Map node IDs to grid positions
            for node_id in active_junctions:
                # Convert linear index to 2D grid coordinates
                row = node_id // cols
                col = node_id % cols
                # Use negative row to flip y-axis (standard grid orientation)
                pos[node_id] = (col, -row)
        else:
            # Use spring layout as fallback
            pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Calculate A matrix for simulation
        A_total = self.sum_matrices()
        initial_densities = [road.density for road in self.roads]
        
        # Store data for plotting
        step_data = []
        total_densities = []
        
        def update_frame(frame_num):
            ax1.clear()
            ax2.clear()
            
            # Step simulation
            if frame_num > 0:
                current_vector = self.get_current_vector()
                self.step_forward(A_total, current_vector)
            
            # Calculate total density for plot
            total_density = sum(road.density for road in self.roads)
            total_densities.append(total_density)
            
            # Network visualization (left plot)
            if G.nodes():
                # Get road densities for edge colors
                edge_colors = []
                edge_labels = {}
                for edge in G.edges():
                    if edge in edge_road_mapping:
                        road = edge_road_mapping[edge]
                        edge_labels[edge] = f"{road.density:.0f}"
                        
                        # Color intensity based on density
                        max_density = max(r.density for r in self.roads) if self.roads else 1
                        color_intensity = min(road.density / max_density, 1.0) if max_density > 0 else 0
                        edge_colors.append(plt.cm.Reds(color_intensity))
                    else:
                        edge_colors.append('gray')
                
                # Draw network with improved visibility
                nx.draw_networkx_nodes(G, pos, ax=ax1, node_color='lightblue', 
                                     node_size=300, alpha=0.8)
                nx.draw_networkx_edges(G, pos, ax=ax1, edge_color=edge_colors, 
                                     width=1.5, arrowsize=8, arrowstyle='->', alpha=0.7)
                
                # Only show node labels for smaller networks to avoid clutter
                if len(G.nodes()) <= 100:
                    nx.draw_networkx_labels(G, pos, {node: f"J{node}" for node in G.nodes()}, 
                                          ax=ax1, font_size=6, font_weight='bold')
                
                # Show edge labels only for very small networks
                if len(edge_labels) <= 50:
                    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax1, font_size=4, 
                                               bbox=dict(boxstyle='round,pad=0.1', 
                                                       facecolor='yellow', alpha=0.5))
                
                ax1.set_title(f"Manhattan Grid Traffic (Step {frame_num})\n{len(G.nodes())} junctions, {len(G.edges())} roads", 
                            fontsize=12)
                ax1.set_aspect('equal')
                ax1.grid(True, alpha=0.3)
                
                # Add color bar legend
                if frame_num == 0:  # Only add once
                    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, 
                                             norm=plt.Normalize(vmin=0, vmax=max_density))
                    sm.set_array([])
                    cbar = fig.colorbar(sm, ax=ax1, shrink=0.6)
                    #cbar.set_label('Traffic Density', rotation=270, labelpad=15)
            
            # Total density plot (right plot)
            if total_densities:
                ax2.plot(range(len(total_densities)), total_densities, 'b-', linewidth=2)
                ax2.scatter(len(total_densities)-1, total_densities[-1], color='red', s=50, zorder=5)
                ax2.set_xlabel('Time Step')
                #ax2.set_ylabel('Total Traffic Density')
                ax2.set_title('Total Network Density Over Time')
                ax2.grid(True, alpha=0.3)
                ax2.text(0.02, 0.98, f"Current: {total_density:.2f}", transform=ax2.transAxes, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), 
                        fontsize=10, verticalalignment='top')
        
        # Create animation
        ani = animation.FuncAnimation(fig, update_frame, frames=num_steps, 
                                    interval=interval, repeat=False)
        
        # Save animation
        print(f"Saving animation to {timestamped_filename}...")
        try:
            ani.save(timestamped_filename, writer='pillow', fps=1000//interval)
            print(f"Animation saved successfully as {timestamped_filename}")
        except Exception as e:
            print(f"Error saving animation: {e}")
            # Try alternative method
            try:
                mp4_filename = timestamped_filename.replace('.gif', '.mp4')
                ani.save(mp4_filename, writer='ffmpeg', fps=1000//interval)
                print(f"Animation saved as MP4 instead: {mp4_filename}")
            except:
                print("Could not save animation. Running simulation and showing final results...")
        
        # Reset to initial state
        for i, density in enumerate(initial_densities):
            self.roads[i].density = density
        
        plt.close()
        return ani
    
    def check_mass_conservation(self, initial_total_density):
        """
        Check if total mass is conserved in the system.
        
        Args:
            initial_total_density: The initial total density to compare against
            
        Returns:
            tuple: (current_total, difference, is_conserved)
        """
        current_total = sum(road.density for road in self.roads)
        difference = abs(current_total - initial_total_density)
        is_conserved = difference < 1e-10  # Allow for floating point precision errors
        
        return current_total, difference, is_conserved
    
    def step_forward_with_conservation_check(self, A_total, current_vector, check_conservation=True):
        """
        Step forward with optional mass conservation checking.
        """
        if check_conservation:
            initial_total = sum(road.density for road in self.roads)
        
        # Perform the step
        result = self.step_forward(A_total, current_vector)
        
        if check_conservation:
            current_total, difference, is_conserved = self.check_mass_conservation(initial_total)
            if not is_conserved:
                print(f"WARNING: Mass not conserved! Initial: {initial_total:.6f}, Current: {current_total:.6f}, Difference: {difference:.6f}")
            else:
                print(f"Mass conserved: {current_total:.6f} (diff: {difference:.2e})")
        
        return result
    
    def analyze_numerical_precision(self, A_total, current_vector):
        """
        Analyze potential sources of numerical precision errors in the simulation.
        """
        print("=== NUMERICAL PRECISION ANALYSIS ===")
        
        # Check matrix condition number
        try:
            cond_number = np.linalg.cond(A_total)
            print(f"A_total matrix condition number: {cond_number:.2e}")
            if cond_number > 1e12:
                print("⚠️  WARNING: Matrix is poorly conditioned (may cause numerical errors)")
        except:
            print("Could not compute condition number")
        
        # Check for very small or very large values
        v_delta = A_total @ current_vector
        print(f"v_delta range: [{np.min(v_delta):.2e}, {np.max(v_delta):.2e}]")
        print(f"current_vector range: [{np.min(current_vector):.2e}, {np.max(current_vector):.2e}]")
        
        # Check matrix sparsity
        nonzero_elements = np.count_nonzero(A_total)
        total_elements = A_total.size
        sparsity = 1 - nonzero_elements / total_elements
        print(f"Matrix sparsity: {sparsity:.1%} ({nonzero_elements}/{total_elements} non-zero)")
        
        # Check for acceptance rate precision issues
        acceptance_rates = [road.get_acceptance_rate() for road in self.roads]
        print(f"Acceptance rates range: [{min(acceptance_rates):.2e}, {max(acceptance_rates):.2e}]")
        
        # Check data types
        print(f"A_total dtype: {A_total.dtype}")
        print(f"current_vector dtype: {current_vector.dtype}")
        print(f"Road densities dtype: {type(self.roads[0].density) if self.roads else 'N/A'}")
        
        print("=====================================")
