import csv
import os
import networkx as nx

# Paths to input files
SCATS_TO_NODE_PATH = 'scats_to_node_mapping.csv'
NODE_DATA_PATH = 'node_data.csv'
EDGE_DATA_PATH = 'edges_data.csv'
MAX_PATH_LENGTH = 10

# Output paths
SCAT_GRAPH_DIR = 'scat-graph'
SCAT_NODE_DATA_PATH = os.path.join(SCAT_GRAPH_DIR, 'node_data.csv')
SCAT_EDGE_DATA_PATH = os.path.join(SCAT_GRAPH_DIR, 'edge_data.csv')

os.makedirs(SCAT_GRAPH_DIR, exist_ok=True)

def load_scats_to_node(path):
    mapping = {}
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row['SiteID']] = row['Node_osmid']
    return mapping

def load_nodes(path):
    nodes = set()
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nodes.add(row['osmid'])
    return nodes

def load_edges(path):
    edges = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append((row['u'], row['v']))
    return edges

def build_graph(nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

def main():
    scats_to_node = load_scats_to_node(SCATS_TO_NODE_PATH)
    nodes = load_nodes(NODE_DATA_PATH)
    edges = load_edges(EDGE_DATA_PATH)
    G = build_graph(nodes, edges)

    # Only keep nodes assigned to SCATS sensors
    scat_nodes = set(scats_to_node.values())
    scat_subgraph = nx.Graph()
    scat_subgraph.add_nodes_from(scat_nodes)

    # Efficient path finding using BFS up to MAX_PATH_LENGTH
    from collections import deque
    pair_paths = []
    for start in scat_nodes:
        visited = {start: [start]}
        queue = deque([(start, [start])])
        while queue:
            current, path = queue.popleft()
            if len(path) - 1 >= MAX_PATH_LENGTH:
                continue
            for neighbor in G.neighbors(current):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    visited[neighbor] = new_path
                    queue.append((neighbor, new_path))
        for end in scat_nodes:
            if end != start and end in visited:
                path = visited[end]
                pair_paths.append((start, end, path, len(path)-1))

    # Remove duplicate pairs (keep shortest path)
    pair_dict = {}
    for n1, n2, path, plen in pair_paths:
        key = tuple(sorted([n1, n2]))
        if key not in pair_dict or plen < pair_dict[key][1]:
            pair_dict[key] = (path, plen)
    pair_paths = [(key[0], key[1], val[0], val[1]) for key, val in pair_dict.items()]
    pair_paths.sort(key=lambda x: x[3])

    used_edges = set()
    for n1, n2, path, plen in pair_paths:
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        if any(e in used_edges or (e[1], e[0]) in used_edges for e in path_edges):
            continue
        for u, v in path_edges:
            scat_subgraph.add_edge(u, v)
            used_edges.add((u, v))

    # Write nodes
    with open(SCAT_NODE_DATA_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['node_id'])
        for node in scat_subgraph.nodes:
            writer.writerow([node])

    # Write edges
    with open(SCAT_EDGE_DATA_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['from_node', 'to_node'])
        for u, v in scat_subgraph.edges:
            writer.writerow([u, v])

if __name__ == '__main__':
    main()
