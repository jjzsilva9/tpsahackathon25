import numpy as np

class Road:
    # 
    def __init__(self, start, end, id, capacity, name=None, occ=0, details=None):
        self.start = start
        self.end = end
        self.id = id
        self.name = name
        self.occ = occ  # Use the parameter value instead of hardcoding to 0
        self.capacity = capacity
        self.details = details
        return

    #Returns True (False) if full (not full)
    def is_full(self):
        return self.occ >= self.capacity
    
    def update_occ(self,new_occ):
        self.occ = new_occ
        return
    
    def __str__(self):
        return f"Road({self.start}->{self.end}, cars={self.occ}/{self.capacity})"
        
class Junction:
    def __init__(self, id, name=None):
        self.id = id
        self.name = name
        self.connected_roads = set()  # Changed to set for easier road management
        return

    def __repr__(self):
        return f"Junction({self.id})"

class Network:
    def __init__(self):
        self.junctions = []
        self.roads = []
        pass

    def add_junction(self, id, name = None):
        self.junctions.append(Junction(id,name=name))
        return

    def initialise_junctions(self, num_junctions):
        for i in range(num_junctions):
            self.junctions.append(Junction(id=i))

    def add_road(self, start_id, end_id, capacity, occ=0, name=None):
        road_id = f'J{start_id}J{end_id}'
        self.roads.append(Road(start_id, end_id, road_id, capacity, name=name, occ=occ))
        
        # Connect the road to both junctions
        self.junctions[start_id].connected_roads.add(road_id)
        self.junctions[end_id].connected_roads.add(road_id)


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
                        occ=0, 
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
            road_count = len(junction.connected_roads)
            print(f"  J{junction.id}: {junction_name} ({road_count} roads)")
        
        print()
        
        # Display road summary
        print(f"🛣️  ROADS ({len(self.roads)} total)")
        print("-" * 50)
        
        if not self.roads:
            print("  No roads in the network.")
        else:
            for road in self.roads:
                # Calculate occ percentage
                occ_percent = (road.occ / road.capacity * 100) if road.capacity > 0 else 0
                
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
                    capacity_info = f"[{road.occ}/{road.capacity}] {status}"
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
                adj_matrix[road.start][road.end] = road.capacity
            
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

    def calculate_A(self, junc_id):
        num_connected_roads = len(self.junctions[junc_id].connected_roads)
        A = np.zeros((num_connected_roads,num_connected_roads))
        return A
    
    def step_forward(self, A_total):
        v_old = np.array([road.occ for road in self.roads])
        v_new = A_total @ v_old
        for idx,occ in enumerate(v_new):
            self.roads[idx].occ = occ
        pass


