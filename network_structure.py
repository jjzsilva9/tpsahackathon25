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

