import csv
import os
import pandas as pd

def main():
    # Paths
    SCATS_TO_NODE_PATH = 'scats_to_node_mapping.csv'
    EDGES_DATA_PATH = 'edges_data.csv'
    SCATS_VOLUME_PATH = 'scats-data/SCATSJanuary2023.csv'
    OUTPUT_PATH = 'edge_density_timeseries.csv'

    # Read SCATS to node mapping
    scats_map = {}
    with open(SCATS_TO_NODE_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scats_map[row['Node_osmid']] = row['SiteID']

    # Read edges and lengths
    edges = []
    edge_lengths = {}
    with open(EDGES_DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            u, v = row['u'], row['v']
            edges.append((u, v))
            edge_lengths[(u, v)] = float(row['length']) if row['length'] else None

    # Read SCATS volume data (timestamped)
    df = pd.read_csv(SCATS_VOLUME_PATH)
    # Columns: Site, End_Time, Avg_Volume (or Sum_Volume)
    if 'End_Time' not in df.columns:
        raise ValueError('End_Time column not found in SCATS volume data.')
    if 'Avg_Volume' in df.columns:
        volume_col = 'Avg_Volume'
    elif 'Sum_Volume' in df.columns:
        volume_col = 'Sum_Volume'
    else:
        raise ValueError('No volume column found in SCATS volume data.')

    # Build a lookup: (Site, End_Time) -> Volume
    volume_lookup = {}
    for _, row in df.iterrows():
        site_id = str(row['Site'])
        ts = row['End_Time']
        vol = row[volume_col]
        volume_lookup[(site_id, ts)] = vol

    # Get all timestamps
    timestamps = sorted(df['End_Time'].unique())

    # Get outgoing edge counts for each node
    outgoing_counts = {}
    for u, v in edges:
        outgoing_counts[u] = outgoing_counts.get(u, 0) + 1
        outgoing_counts[v] = outgoing_counts.get(v, 0) + 0  # Only count outgoing from u

    # Prepare output
    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'from_node', 'to_node', 'density'])
        for ts in timestamps:
            for u, v in edges:
                # Get SiteIDs for nodes
                site_u = scats_map.get(str(u))
                site_v = scats_map.get(str(v))
                # Get volumes for this timestamp
                vol_u = volume_lookup.get((site_u, ts), None)
                vol_v = volume_lookup.get((site_v, ts), None)
                out_u = outgoing_counts.get(u, 1)
                out_v = outgoing_counts.get(v, 1)
                # Estimate flow for edge
                flow_u = float(vol_u) / out_u if vol_u and out_u else None
                flow_v = float(vol_v) / out_v if vol_v and out_v else None
                if flow_u and flow_v:
                    flow = (flow_u + flow_v) / 2
                elif flow_u:
                    flow = flow_u
                elif flow_v:
                    flow = flow_v
                else:
                    flow = None
                # Calculate density
                length = edge_lengths.get((u, v), None)
                density = None
                if flow and length:
                    try:
                        density = flow / (length / 1000)  # vehicles per km
                    except Exception:
                        density = None
                writer.writerow([ts, u, v, density if density is not None else ''])

if __name__ == '__main__':
    main()
