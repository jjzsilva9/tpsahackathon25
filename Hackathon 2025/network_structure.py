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
                total_density += density
                total_capacity += capacity
                
                fill_percent = (density / capacity * 100) if capacity > 0 else 0
                
                # Status description
                if fill_percent == 0:
                    status = "Empty"
                elif fill_percent < 25:
                    status = "Light Traffic"
                elif fill_percent < 50:
                    status = "Moderate Traffic"
                elif fill_percent < 75:
                    status = "Heavy Traffic"
                else:
                    status = "Very Heavy/Full"
                
                print(f"  Road J{road.start}→J{road.end}:")
                print(f"    Density: {density:.2f}/{capacity} ({fill_percent:.1f}%)")
                print(f"    Status: {status}")
                print(f"    Potential: {road.potential:.2f}")
                print()
            
            overall_percent = (total_density / total_capacity * 100) if total_capacity > 0 else 0
            print(f"NETWORK SUMMARY:")
            print(f"  Total Density: {total_density:.2f}")
            print(f"  Total Capacity: {total_capacity}")
            print(f"  Overall Fill: {overall_percent:.1f}%")
            print("-" * 60)

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
        v_delta = A_total @ current_vector

        for road in self.roads:
            road.update_occ(road.density + v_delta[self.get_road_index(road.id)])
        
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
                        
                        # Color intensity based on density with better contrast
                        max_density = max(r.density for r in self.roads) if self.roads else 1
                        min_density = min(r.density for r in self.roads) if self.roads else 0
                        
                        # Normalize with better contrast (0.2 to 1.0 range instead of 0 to 1)
                        if max_density > min_density:
                            normalized_density = (road.density - min_density) / (max_density - min_density)
                            color_intensity = 0.2 + 0.8 * normalized_density  # Scale to 0.2-1.0
                        else:
                            color_intensity = 0.5
                        
                        edge_colors.append(plt.cm.Reds(color_intensity))
                    else:
                        edge_colors.append('gray')
                
                # Draw network with improved visibility
                nx.draw_networkx_nodes(G, pos, ax=ax1, node_color='lightblue', 
                                     node_size=150, alpha=0.8)
                nx.draw_networkx_edges(G, pos, ax=ax1, edge_color=edge_colors, 
                                     width=3, arrowsize=12, arrowstyle='->', alpha=0.9)
                
                # Only show node labels for smaller networks to avoid clutter
                if len(G.nodes()) <= 100:
                    nx.draw_networkx_labels(G, pos, {node: f"J{node}" for node in G.nodes()}, 
                                          ax=ax1, font_size=5, font_weight='bold')
                
                # Show edge labels only for very small networks
                if len(edge_labels) <= 30:
                    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax1, font_size=4, 
                                               bbox=dict(boxstyle='round,pad=0.1', 
                                                       facecolor='yellow', alpha=0.5))
                
                ax1.set_title(f"Manhattan Grid Traffic (Step {frame_num})\n{len(G.nodes())} junctions, {len(G.edges())} roads", 
                            fontsize=12)
                ax1.set_aspect('equal')
                ax1.grid(True, alpha=0.3)
            
            # Total density plot (right plot)
            if total_densities:
                ax2.plot(range(len(total_densities)), total_densities, 'b-', linewidth=2)
                ax2.scatter(len(total_densities)-1, total_densities[-1], color='red', s=50, zorder=5)
                ax2.set_xlabel('Time Step')
                ax2.set_ylabel('Total Traffic Density')
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
