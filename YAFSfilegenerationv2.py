import random
import json
import os
import networkx as nx
import pandas as pd
from networkx.readwrite import json_graph
from random import randrange
from collections import defaultdict

# Define helper functions to create modules, messages, and transmissions
def create_module(id, name, ram, storage):
    return {
        "RAM": ram,
        "Storage": storage,
        "type": "MODULE",
        "id": id,
        "name": name
    }

def create_message(id, name, s, d, bytes, instructions):
    return {
        "d": d,
        "bytes": bytes,
        "name": name,
        "s": s,
        "id": id,
        "instructions": instructions
    }

def create_transmission(message_out=None, module=None, message_in=None):
    transmission = {"module": module}
    if message_out is not None:
        transmission["message_out"] = message_out
    if message_in is not None:
        transmission["message_in"] = message_in
    return transmission


# Define the function to calculate centrality
def hitung_central(degree):
    node = []
    centrality = []
    for i in degree:
        node.append(i)
        centrality.append(degree[i])
    df_centrality = pd.DataFrame(list(zip(node, centrality)), columns=['node_id', 'central'])
    return df_centrality

# # Define the create_json_topology function
# def create_json_topology(m_value, jumlah_node, ram_range, speed_range, storage_range, propagasi, bandwith, json_filename):
#     G = nx.barabasi_albert_graph(jumlah_node, m_value)
#     ls = list(G.nodes)
#     li = {x: int(x) for x in ls}
#     nx.relabel_nodes(G, li, False)

#     # Setting attributes for nodes and edges
#     min_ram, max_ram = ram_range
#     min_speed, max_speed = speed_range
#     min_storage, max_storage = storage_range

#     penentuanBW = {x: bandwith for x in G.edges()}
#     penentuanPR = {x: propagasi for x in G.edges()}
#     penentuanSpeed = {x: randrange(min_speed, max_speed) for x in G.nodes()}
#     penentuanRAM = {x: randrange(min_ram, max_ram) for x in G.nodes()}
#     penentuanStorage = {x: randrange(min_storage, max_storage) for x in G.nodes()}

#     nx.set_node_attributes(G, values=penentuanSpeed, name="Speed")
#     nx.set_node_attributes(G, values=penentuanRAM, name="RAM")
#     nx.set_node_attributes(G, values=penentuanStorage, name="Storage")
#     nx.set_edge_attributes(G, name='BW', values=penentuanBW)
#     nx.set_edge_attributes(G, name='PR', values=penentuanPR)

#     # Calculate node costs based on speed bins
#     costs = {}
#     num_bins = 5
#     cost_values = [1, 5, 10, 15, 20]
#     # max_speed - 1 because randrange excludes the upper bound
#     speed_range_width = (max_speed - 1) - min_speed
#     bin_size = speed_range_width / num_bins if speed_range_width > 0 else 1

#     for node_id in G.nodes():
#         speed = G.nodes[node_id]["Speed"]
#         # Determine the bin index (0 to 4)
#         if bin_size > 0:
#             bin_index = min(int((speed - min_speed) // bin_size), num_bins - 1)
#         else: # Handle case where min_speed == max_speed - 1
#              bin_index = 0
#         costs[node_id] = cost_values[bin_index]

#     # Set node costs
#     nx.set_node_attributes(G, values=costs, name="Cost")

#     # Calculate centrality and sort nodes
#     dc = nx.betweenness_centrality(G)
#     df_centrality = hitung_central(dc)
#     sorted_centrality = df_centrality.sort_values(by='central', ascending=False)

#   # Calculate the size of a quarter
#     quarter_size = jumlah_node // 4

#     # Label the top and bottom quarters differently
#     top_quarter = sorted_centrality.head(quarter_size)['node_id']
#     bottom_quarter = sorted_centrality.tail(quarter_size)['node_id']

#     # Label nodes as "node head", "gateway", or "fog node"
#     for node in G.nodes():
#         if node in top_quarter:
#             G.nodes[node]['label'] = 'head_node'
#         elif node in bottom_quarter:
#             G.nodes[node]['label'] = 'gateway'
#         else:
#             G.nodes[node]['label'] = 'fog_node'

#     # Add a special "cloud" node
#     cloud_node_id = jumlah_node
#     G.add_node(cloud_node_id)
#     G.nodes[cloud_node_id]['Speed'] = 10000
#     G.nodes[cloud_node_id]['RAM'] = 999999999
#     G.nodes[cloud_node_id]['Storage'] = 999999999
#     G.nodes[cloud_node_id]['Cost'] = 10
#     G.nodes[cloud_node_id]['label'] = 'cloud'

#     # Connect the "cloud" node to "node head" nodes
#     for node_id in top_quarter:
#         G.add_edge(node_id, cloud_node_id)
#         G[node_id][cloud_node_id]['BW'] = 150000
#         G[node_id][cloud_node_id]['PR'] = 100

#     # Check if the file exists, if it does not exist, save the graph to JSON
#     # Ensure the directory exists before trying to write the file
#     os.makedirs(os.path.dirname(json_filename), exist_ok=True)
    
#     if not os.path.exists(json_filename): # This check might be redundant now if overwriting is acceptable
#         with open(json_filename, 'w') as outfile:
#             outfile.write(json.dumps(json_graph.node_link_data(G)))
#     else:
#         print("cannot save in that path, please add new path.")

    
#     return print(f"done on generating on {json_filename}")


# Define the create_json_topology function
def create_json_topology(m_value, jumlah_node, ram_range, speed_range, storage_range, propagasi, bandwith, json_filename):
    G = nx.barabasi_albert_graph(jumlah_node, m_value)
    ls = list(G.nodes)
    li = {x: int(x) for x in ls}
    nx.relabel_nodes(G, li, False)

    # Setting attributes for nodes and edges
    min_ram, max_ram = ram_range
    min_speed, max_speed = speed_range
    min_storage, max_storage = storage_range

    penentuanBW = {x: bandwith for x in G.edges()}
    penentuanPR = {x: propagasi for x in G.edges()}
    penentuanSpeed = {x: randrange(min_speed, max_speed) for x in G.nodes()}
    penentuanRAM = {x: randrange(min_ram, max_ram) for x in G.nodes()}
    penentuanStorage = {x: randrange(min_storage, max_storage) for x in G.nodes()}

    nx.set_node_attributes(G, values=penentuanSpeed, name="Speed")
    nx.set_node_attributes(G, values=penentuanRAM, name="RAM")
    nx.set_node_attributes(G, values=penentuanStorage, name="Storage")
    nx.set_edge_attributes(G, name='BW', values=penentuanBW)
    nx.set_edge_attributes(G, name='PR', values=penentuanPR)
    
    # Calculate node costs based on speed bins
    costs = {}
    num_bins = 5
    cost_values = [1, 5, 10, 15, 20]
    # max_speed - 1 because randrange excludes the upper bound
    speed_range_width = (max_speed - 1) - min_speed
    bin_size = speed_range_width / num_bins if speed_range_width > 0 else 1

    for node_id in G.nodes():
        speed = G.nodes[node_id]["Speed"]
        # Determine the bin index (0 to 4)
        if bin_size > 0:
            bin_index = min(int((speed - min_speed) // bin_size), num_bins - 1)
        else: # Handle case where min_speed == max_speed - 1
             bin_index = 0
        costs[node_id] = cost_values[bin_index]

    # Set node costs
    nx.set_node_attributes(G, values=costs, name="Cost")

    # Calculate centrality and sort nodes
    dc = nx.betweenness_centrality(G)
    df_centrality = hitung_central(dc)
    sorted_centrality = df_centrality.sort_values(by='central', ascending=False)

  # Calculate the size of a quarter
    quarter_size = jumlah_node // 4

    # Label the top and bottom quarters differently
    top_quarter = sorted_centrality.head(quarter_size)['node_id']
    bottom_quarter = sorted_centrality.tail(quarter_size)['node_id']

    # Label nodes as "node head", "gateway", or "fog node"
    for node in G.nodes():
        if node in top_quarter:
            G.nodes[node]['label'] = 'head_node'
        elif node in bottom_quarter:
            G.nodes[node]['label'] = 'gateway'
        else:
            G.nodes[node]['label'] = 'fog_node'

    # Add a special "cloud" node
    cloud_node_id = jumlah_node
    G.add_node(cloud_node_id)
    G.nodes[cloud_node_id]['Speed'] = 10000
    G.nodes[cloud_node_id]['RAM'] = 999999999
    G.nodes[cloud_node_id]['Storage'] = 999999999
    G.nodes[cloud_node_id]['Cost'] = 10
    G.nodes[cloud_node_id]['label'] = 'cloud'

    # Connect the "cloud" node to "node head" nodes
    for node_id in top_quarter:
        G.add_edge(node_id, cloud_node_id)
        G[node_id][cloud_node_id]['BW'] = 125000
        G[node_id][cloud_node_id]['PR'] = 100

    # Check if the file exists, if it does not exist, save the graph to JSON
    # Ensure the directory exists before trying to write the file
    os.makedirs(os.path.dirname(json_filename), exist_ok=True)
    
    if not os.path.exists(json_filename): # This check might be redundant now if overwriting is acceptable
        with open(json_filename, 'w') as outfile:
            outfile.write(json.dumps(json_graph.node_link_data(G)))
    else:
        print("cannot save in that path, please add new path.")

    
    return print(f"done on generating on {json_filename}")


# Define the create_json_topology function
def create_json_topology_withlinktype(m_value, jumlah_node, ram_range, speed_range, storage_range, propagasi, bandwith, json_filename):
    G = nx.barabasi_albert_graph(jumlah_node, m_value)
    ls = list(G.nodes)
    li = {x: int(x) for x in ls}
    nx.relabel_nodes(G, li, False)

    # Setting attributes for nodes and edges
    min_ram, max_ram = ram_range
    min_speed, max_speed = speed_range
    min_storage, max_storage = storage_range

    penentuanBW = {x: bandwith for x in G.edges()}
    penentuanPR = {x: propagasi for x in G.edges()}
    penentuanSpeed = {x: randrange(min_speed, max_speed) for x in G.nodes()}
    penentuanRAM = {x: randrange(min_ram, max_ram) for x in G.nodes()}
    penentuanStorage = {x: randrange(min_storage, max_storage) for x in G.nodes()}

    nx.set_node_attributes(G, values=penentuanSpeed, name="Speed")
    nx.set_node_attributes(G, values=penentuanRAM, name="RAM")
    nx.set_node_attributes(G, values=penentuanStorage, name="Storage")
    nx.set_edge_attributes(G, name='BW', values=penentuanBW)
    nx.set_edge_attributes(G, name='PR', values=penentuanPR)
    
    # Calculate node costs based on speed bins
    costs = {}
    num_bins = 5
    cost_values = [1, 5, 10, 15, 20]
    # max_speed - 1 because randrange excludes the upper bound
    speed_range_width = (max_speed - 1) - min_speed
    bin_size = speed_range_width / num_bins if speed_range_width > 0 else 1

    for node_id in G.nodes():
        speed = G.nodes[node_id]["Speed"]
        # Determine the bin index (0 to 4)
        if bin_size > 0:
            bin_index = min(int((speed - min_speed) // bin_size), num_bins - 1)
        else: # Handle case where min_speed == max_speed - 1
             bin_index = 0
        costs[node_id] = cost_values[bin_index]

    # Set node costs
    nx.set_node_attributes(G, values=costs, name="Cost")

    # Calculate centrality and sort nodes
    dc = nx.betweenness_centrality(G)
    df_centrality = hitung_central(dc)
    sorted_centrality = df_centrality.sort_values(by='central', ascending=False)

  # Calculate the size of a quarter
    quarter_size = jumlah_node // 4

    # Label the top and bottom quarters differently
    top_quarter = sorted_centrality.head(quarter_size)['node_id']
    bottom_quarter = sorted_centrality.tail(quarter_size)['node_id']

    # Label nodes as "node head", "gateway", or "fog node"
    for node in G.nodes():
        if node in top_quarter:
            G.nodes[node]['label'] = 'head_node'
        elif node in bottom_quarter:
            G.nodes[node]['label'] = 'gateway'
        else:
            G.nodes[node]['label'] = 'fog_node'

    # Add a special "cloud" node
    cloud_node_id = jumlah_node
    G.add_node(cloud_node_id)
    G.nodes[cloud_node_id]['Speed'] = 10000
    G.nodes[cloud_node_id]['RAM'] = 999999999
    G.nodes[cloud_node_id]['Storage'] = 999999999
    G.nodes[cloud_node_id]['Cost'] = 10
    G.nodes[cloud_node_id]['label'] = 'cloud'

    # Connect the "cloud" node to "node head" nodes
    for node_id in top_quarter:
        G.add_edge(node_id, cloud_node_id)
        G[node_id][cloud_node_id]['BW'] = 125000
        G[node_id][cloud_node_id]['PR'] = 100

    # Check if the file exists, if it does not exist, save the graph to JSON
    # Ensure the directory exists before trying to write the file
    os.makedirs(os.path.dirname(json_filename), exist_ok=True)
    
    if not os.path.exists(json_filename): # This check might be redundant now if overwriting is acceptable
        with open(json_filename, 'w') as outfile:
            outfile.write(json.dumps(json_graph.node_link_data(G)))
    else:
        print("cannot save in that path, please add new path.")

    


    # Update the link types in the JSON file
    folder_filepath = os.path.dirname(json_filename)
    input_json_filename = os.path.basename(json_filename)
    output_json_filename = f"updatedwithlinks_{input_json_filename}"
    
    updated_filepath = update_link_type_in_json(json_filename, os.path.join(folder_filepath, output_json_filename), further_revised_categorize_link)
    
    print(f"Updated topology with link types saved to: {updated_filepath}")
    summary = process_genetic_network(updated_filepath)
    print_summary(summary)



# Define the create_json_topology function
def create_json_topology_propagasirange(m_value, jumlah_node, ram_range, speed_range, storage_range, propagasi_range, bandwidth_range, json_filename):
    G = nx.barabasi_albert_graph(jumlah_node, m_value)
    ls = list(G.nodes)
    li = {x: int(x) for x in ls}
    nx.relabel_nodes(G, li, False)

    # Setting attributes for nodes and edges
    min_ram, max_ram = ram_range
    min_speed, max_speed = speed_range
    min_storage, max_storage = storage_range
    min_propagasi, max_propagasi = propagasi_range
    min_bandwidth, max_bandwidth = bandwidth_range

    penentuanBW = {x: randrange(min_bandwidth, max_bandwidth) for x in G.edges()}
    penentuanPR = {x: randrange(min_propagasi, max_propagasi) for x in G.edges()}
    penentuanSpeed = {x: randrange(min_speed, max_speed) for x in G.nodes()}
    penentuanRAM = {x: randrange(min_ram, max_ram) for x in G.nodes()}
    penentuanStorage = {x: randrange(min_storage, max_storage) for x in G.nodes()}

    nx.set_node_attributes(G, values=penentuanSpeed, name="Speed")
    nx.set_node_attributes(G, values=penentuanRAM, name="RAM")
    nx.set_node_attributes(G, values=penentuanStorage, name="Storage")
    nx.set_edge_attributes(G, name='BW', values=penentuanBW)
    nx.set_edge_attributes(G, name='PR', values=penentuanPR)
    
    # Calculate node costs based on speed bins
    costs = {}
    num_bins = 5
    cost_values = [1, 5, 10, 15, 20]
    # max_speed - 1 because randrange excludes the upper bound
    speed_range_width = (max_speed - 1) - min_speed
    bin_size = speed_range_width / num_bins if speed_range_width > 0 else 1

    for node_id in G.nodes():
        speed = G.nodes[node_id]["Speed"]
        # Determine the bin index (0 to 4)
        if bin_size > 0:
            bin_index = min(int((speed - min_speed) // bin_size), num_bins - 1)
        else: # Handle case where min_speed == max_speed - 1
             bin_index = 0
        costs[node_id] = cost_values[bin_index]

    # Set node costs
    nx.set_node_attributes(G, values=costs, name="Cost")

    # Calculate centrality and sort nodes
    dc = nx.betweenness_centrality(G)
    df_centrality = hitung_central(dc)
    sorted_centrality = df_centrality.sort_values(by='central', ascending=False)

  # Calculate the size of a quarter
    quarter_size = jumlah_node // 4

    # Label the top and bottom quarters differently
    top_quarter = sorted_centrality.head(quarter_size)['node_id']
    bottom_quarter = sorted_centrality.tail(quarter_size)['node_id']

    # Label nodes as "node head", "gateway", or "fog node"
    for node in G.nodes():
        if node in top_quarter:
            G.nodes[node]['label'] = 'head_node'
        elif node in bottom_quarter:
            G.nodes[node]['label'] = 'gateway'
        else:
            G.nodes[node]['label'] = 'fog_node'

    # Add a special "cloud" node
    cloud_node_id = jumlah_node
    G.add_node(cloud_node_id)
    G.nodes[cloud_node_id]['Speed'] = 10000
    G.nodes[cloud_node_id]['RAM'] = 999999999
    G.nodes[cloud_node_id]['Storage'] = 999999999
    G.nodes[cloud_node_id]['Cost'] = 10
    G.nodes[cloud_node_id]['label'] = 'cloud'

    # Connect the "cloud" node to "node head" nodes
    for node_id in top_quarter:
        G.add_edge(node_id, cloud_node_id)
        G[node_id][cloud_node_id]['BW'] = 125000
        G[node_id][cloud_node_id]['PR'] = 100

    # Check if the file exists, if it does not exist, save the graph to JSON
    # Ensure the directory exists before trying to write the file
    os.makedirs(os.path.dirname(json_filename), exist_ok=True)
    
    if not os.path.exists(json_filename): # This check might be redundant now if overwriting is acceptable
        with open(json_filename, 'w') as outfile:
            outfile.write(json.dumps(json_graph.node_link_data(G)))
    else:
        print("cannot save in that path, please add new path.")

    


    # Update the link types in the JSON file
    folder_filepath = os.path.dirname(json_filename)
    input_json_filename = os.path.basename(json_filename)
    output_json_filename = f"updatedwithlinks_{input_json_filename}"
    
    updated_filepath = update_link_type_in_json(json_filename, os.path.join(folder_filepath, output_json_filename), further_revised_categorize_link)
    
    print(f"Updated topology with link types saved to: {updated_filepath}")
    summary = process_genetic_network(updated_filepath)
    print_summary(summary)



 # New code: Update link types in the JSON file
def further_revised_categorize_link(link, nodes):
    source_label = next((node['label'] for node in nodes if node['id'] == link['source']), None)
    target_label = next((node['label'] for node in nodes if node['id'] == link['target']), None)

    if 'cloud' in [source_label, target_label]:
        return 'cloud_link'
    elif 'head_node' in [source_label, target_label]:
        return 'node_head_link'
    elif 'gateway' in [source_label, target_label]:
        return 'gateway_link'
    elif source_label != 'cloud' and target_label != 'cloud' and source_label != 'gateway' and target_label != 'gateway':
        return 'node_link'
    else:
        return 'other'

def update_link_type_in_json(input_filepath, output_filepath, categorization_func):
    with open(input_filepath, 'r') as file:
        json_data = json.load(file)

    nodes = json_data['nodes']
    for link in json_data['links']:
        link['type'] = further_revised_categorize_link(link, nodes)

    with open(output_filepath, 'w') as file:
        json.dump(json_data, file, indent=4)

    return output_filepath




def process_genetic_network(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    nodes = data['nodes']
    links = data['links']

    # Process nodes
    node_summary = {
        'count': len(nodes),
        'speed_range': [float('inf'), float('-inf')],
        'ram_range': [float('inf'), float('-inf')],
        'storage_range': [float('inf'), float('-inf')],
        'labels': defaultdict(int)
    }

    for node in nodes:
        node_summary['speed_range'][0] = min(node_summary['speed_range'][0], node['Speed'])
        node_summary['speed_range'][1] = max(node_summary['speed_range'][1], node['Speed'])
        node_summary['ram_range'][0] = min(node_summary['ram_range'][0], node['RAM'])
        node_summary['ram_range'][1] = max(node_summary['ram_range'][1], node['RAM'])
        node_summary['labels'][node['label']] += 1

    # Process links
    link_summary = {
        'count': len(links),
        'types': defaultdict(int),
        'bw_range': [float('inf'), float('-inf')],
        'pr_range': [float('inf'), float('-inf')]
    }

    for link in links:
        link_summary['types'][link['type']] += 1
        link_summary['bw_range'][0] = min(link_summary['bw_range'][0], link['BW'])
        link_summary['bw_range'][1] = max(link_summary['bw_range'][1], link['BW'])
        link_summary['pr_range'][0] = min(link_summary['pr_range'][0], link['PR'])
        link_summary['pr_range'][1] = max(link_summary['pr_range'][1], link['PR'])

    return {
        'node_summary': node_summary,
        'link_summary': link_summary
    }

def print_summary(summary):
    print("Network Summary:")
    print("\nNodes:")
    print(f"  Total nodes: {summary['node_summary']['count']}")
    print(f"  Speed range: {summary['node_summary']['speed_range']}")
    print(f"  RAM range: {summary['node_summary']['ram_range']}")
    print("  Node labels:")
    for label, count in summary['node_summary']['labels'].items():
        print(f"    {label}: {count}")

    print("\nLinks:")
    print(f"  Total links: {summary['link_summary']['count']}")
    print(f"  Bandwidth range: {summary['link_summary']['bw_range']}")
    print(f"  Propagation delay range: {summary['link_summary']['pr_range']}")
    print("  Link types:")
    for link_type, count in summary['link_summary']['types'].items():
        print(f"    {link_type}: {count}")


def nodes_extract_to_dataframe(filepath):

    try:
        # Reading and parsing the JSON file
        with open(filepath, 'r') as file:
            data = json.load(file)

        # Extracting the 'nodes' data
        nodes_data = data['nodes']

        # Creating a DataFrame from the nodes data
        return pd.DataFrame(nodes_data)
    except Exception as e:
        # In case of any error, return the error message
        return str(e)
    

def links_extract_to_dataframe(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Check if 'links' key exists and is a list
        if 'links' in data and isinstance(data['links'], list):
            flattened_data = pd.json_normalize(data['links'])
            return flattened_data
        else:
            return pd.DataFrame()  # Return an empty DataFrame if 'links' is not found
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error



def analyze_topology(filepath):
    """
    Updated function to analyze the nodes and links in a YAFS topology JSON file,
    with flattened node type counts.
    
    :param filepath: Path to the JSON file.
    :return: Two DataFrames with analysis results for nodes and links.
    """
    try:
        # Extracting nodes and links data
        nodes_df = nodes_extract_to_dataframe(filepath)
        links_df = links_extract_to_dataframe(filepath)

        # Constants
        cloud_node_label = 'cloud'

        # NODE ANALYSIS
        # Identifying cloud nodes
        cloud_nodes_ids = nodes_df[nodes_df['label'] == cloud_node_label]['id'].tolist()
        non_cloud_nodes_df = nodes_df[~nodes_df['id'].isin(cloud_nodes_ids)]

        # Analysis 1: Average RAM on nodes, excluding cloud
        average_ram = non_cloud_nodes_df['RAM'].mean()

        # Analysis 2: Count of types of nodes (flattened)
        node_types_count = nodes_df['label'].value_counts()
        node_types_count_flattened = node_types_count.to_dict()

        # Analysis 3: Total RAM on nodes, excluding cloud
        total_ram = non_cloud_nodes_df['RAM'].sum()

        # Preparing the nodes results DataFrame with flattened node type counts
        nodes_results_data = {
            'Node Analysis': ['Average RAM (excluding cloud)', 'Total RAM (excluding cloud)'] + list(node_types_count_flattened.keys()),
            'Result': [average_ram, total_ram] + list(node_types_count_flattened.values())
        }
        nodes_results = pd.DataFrame(nodes_results_data)

        # LINK ANALYSIS
        # Rest of the link analysis as before
        # ...

        # Preparing the links results DataFrame
        # ...

        # LINK ANALYSIS
        # Analysis 1: Count of links and unique links
        links_df['ordered_pair'] = links_df.apply(lambda row: tuple(sorted([row['source'], row['target']])), axis=1)
        total_links = len(links_df)
        unique_links = len(links_df['ordered_pair'].unique())

        # Counting links to cloud nodes
        count_links_to_cloud = len(links_df[links_df['source'].isin(cloud_nodes_ids) | links_df['target'].isin(cloud_nodes_ids)])

        # Analysis 2 and 3: Average propagation delay and bandwidth excluding cloud
        links_excl_cloud = links_df[~links_df['source'].isin(cloud_nodes_ids) & ~links_df['target'].isin(cloud_nodes_ids)]
        avg_prop_delay_excl_cloud = links_excl_cloud['PR'].mean()
        avg_bandwidth_excl_cloud = links_excl_cloud['BW'].mean()

        # Analysis 4 and 5: Average propagation delay and bandwidth only to cloud
        links_to_cloud = links_df[links_df['source'].isin(cloud_nodes_ids) | links_df['target'].isin(cloud_nodes_ids)]
        avg_prop_delay_to_cloud = links_to_cloud['PR'].mean() if not links_to_cloud.empty else 'No cloud links'
        avg_bandwidth_to_cloud = links_to_cloud['BW'].mean() if not links_to_cloud.empty else 'No cloud links'

        # Preparing the links results DataFrame
        links_results_data = {
            'Link Analysis': [
                'Total Links', 'Unique Links', 'Links to Cloud Nodes',
                'Average Propagation Delay (excluding cloud)', 
                'Average Bandwidth (excluding cloud)', 
                'Average Propagation Delay (to cloud)', 
                'Average Bandwidth (to cloud)'
            ],
            'Result': [
                total_links, unique_links, count_links_to_cloud,
                avg_prop_delay_excl_cloud, 
                avg_bandwidth_excl_cloud, 
                avg_prop_delay_to_cloud, 
                avg_bandwidth_to_cloud
            ]
        }
        links_results = pd.DataFrame(links_results_data)

        return nodes_results, links_results

    except Exception as e:
        return str(e), str(e)




# Adjust the create_application_specs function to ensure the first message is "in" to Mod0 and "out" as M0
def create_application_specs(num_apps, deadline, ram_range, storage_range, module_range,byte_range, instruction_range, safe_path):
    app_specs = []
    for app_id in range(num_apps):
        # the number of module only can be specified here. sorry
        num_modules = random.randint(*module_range)
        modules = [create_module(id=i, name=f"Mod{i}", ram=random.randint(*ram_range), storage=random.randint(*storage_range)) for i in range(num_modules)]
        messages = [create_message(id=i, name=f"M{i}" if i > 0 else f"M.USER.APP.{app_id}",
                                   s=f"Mod{i-1}" if i > 0 else "None",
                                   d=f"Mod{i}" if i < num_modules-1 else "None",
                                   bytes=random.randint(*byte_range),
                                   instructions=random.randint(*instruction_range))
                    for i in range(num_modules)]
        messages[-1]["s"] = f"Mod{num_modules-2}"  # Make the last message come from the second to last module
        messages[-1]["d"] = f"Mod{num_modules-1}"  # Make the last message go to the last module
        transmissions = [create_transmission(message_out=f"M{i+1}", module=f"Mod{i}", message_in=f"M{i}" if i > 0 else f"M.USER.APP.{app_id}")
                         if i < num_modules - 1 
                         else create_transmission(module=f"Mod{i}", message_in=f"M{i}" if i > 0 else f"M.USER.APP.{app_id}")
                         for i in range(num_modules)]
        app_specs.append({
            "name": str(app_id),
            "transmission": transmissions,
            "numberofmodule" : num_modules,
            "module": modules,
            "deadline": random.randint(*deadline),
            "message": messages,
            "id": app_id
        })

    if not os.path.exists(safe_path):
        with open(safe_path, 'w') as file:
            json.dump(app_specs, file, indent=4)
    else:
        print("cannot save in that path, please add new path.")

    return app_specs

def app_analyze_aggregated(json_file_path):
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Prepare an empty list to store rows
    rows = []

    # Iterate over each application
    for app in data:
        app_id = int(app['id'])
        numberofmodule = int(app['numberofmodule'])

        total_ram = sum(module['RAM'] for module in app['module'])
        total_storage = sum(module['Storage'] for module in app['module'])
        total_bytes = sum(message['bytes'] for message in app['message'])
        total_instructions = sum(message['instructions'] for message in app['message'])

        row = {
            'Application ID': app_id,
            'Number of Modules': numberofmodule,
            'Total RAM': total_ram,
            'Total Storage': total_storage,
            'Total Bytes': total_bytes,
            'Total Instructions': total_instructions
        }
        rows.append(row)

    # Convert the list of rows into a DataFrame
    df = pd.DataFrame(rows)

    return df

def module_json_to_dataframe(json_file_path):
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Prepare an empty list to store rows
    rows = []

    # Iterate over each application
    for app in data:
        app_id = app['id']
        numberofmodule = app['numberofmodule']

        # Iterate over each module and message
        for module, message in zip(app['module'], app['message']):
            row = {
                'Application ID': str(app_id),
                'numberofmodule': numberofmodule,
                'Module ID': module['id'],
                'Module Name': module['name'],
                'Required RAM': module['RAM'],
                'Required Storage': module['Storage'],
                'Message ID': message['id'],
                'Message Name': message['name'],
                'Bytes': message['bytes'],
                'Instructions': message['instructions']
            }
            rows.append(row)

    # Convert the list of rows into a DataFrame
    df = pd.DataFrame(rows)

    return df

def create_population_json(app_json_path, topology_json_path, lambda_range, save_json_path):
    # Load the application data
    app_df = app_analyze_aggregated(app_json_path)

    # Load the topology data and filter for gateways
    topology_df = nodes_extract_to_dataframe(topology_json_path)
    gateway_nodes = topology_df[topology_df['label'] == 'gateway']

    # Prepare the population data structure
    population_data = {'sources': []}

    # Randomly allocate sources for each application
    for _, app_row in app_df.iterrows():
        app_id = app_row['Application ID']
        # Select a random gateway node
        selected_node = gateway_nodes.sample().iloc[0]
        # Generate a random lambda value within the specified range
        lambda_value = random.randint(*lambda_range)
        # Construct the source data
        source_data = {
            'id_resource': int(selected_node['id']),  # Convert to standard Python int
            'app': str(app_id),
            'message': f'M.USER.APP.{app_id}',
            'lambda': lambda_value
        }   
        # Add to the population data
        population_data['sources'].append(source_data)

    # Save the population data to a JSON file
    with open(save_json_path, 'w') as f:
        json.dump(population_data, f, indent=4)

    return f"Population data saved to {save_json_path}"

# Function to convert placement JSON to a DataFrame
def placement_json_to_dataframe(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    rows = []
    for placement in data:
        rows.append({
            'Module Name': placement['module_name'],
            'Application ID': str(placement['app']),
            'Node ID': placement['id_resource']
        })
    return pd.DataFrame(rows)

# Main function to check placement validity
def check_placement_validity(topology_file, application_file, placement_file):
    # Parse the JSON files into DataFrames
    nodes_df = nodes_extract_to_dataframe(topology_file)
    application_df = module_json_to_dataframe(application_file)
    placement_df = placement_json_to_dataframe(placement_file)

    
    # # # Diagnostic prints to check DataFrame structures
    # print("Placement DataFrame Columns:", placement_df.columns)
    # print("Application DataFrame Columns:", application_df.columns)
    # print("Data types in Placement DataFrame:", placement_df.dtypes)
    # print("Data types in Application DataFrame:", application_df.dtypes)

    # Merge and calculate RAM and Storage usage
    placement_app_merged = pd.merge(placement_df, application_df, on=['Module Name', 'Application ID'], how='left')
    # Group by Node ID and sum required resources
    node_resource_usage = placement_app_merged.groupby('Node ID')[['Required RAM', 'Required Storage']].sum().reset_index()

    # Rename node capacity columns for clarity
    nodes_df.rename(columns={'RAM': 'Available RAM', 'Storage': 'Available Storage'}, inplace=True)

    # Compare available and required resources
    node_resource_comparison = pd.merge(nodes_df, node_resource_usage, left_on='id', right_on='Node ID', how='left')
    # Fill NaN for nodes with no modules placed (means 0 usage)
    node_resource_comparison[['Required RAM', 'Required Storage']].fillna(0, inplace=True)

    # Check sufficiency
    node_resource_comparison['RAM_Sufficient'] = node_resource_comparison['Available RAM'] >= node_resource_comparison['Required RAM']
    node_resource_comparison['Storage_Sufficient'] = node_resource_comparison['Available Storage'] >= node_resource_comparison['Required Storage']

    # Check if all nodes have sufficient resources and print a message
    if node_resource_comparison['RAM_Sufficient'].all() and node_resource_comparison['Storage_Sufficient'].all():
        print("All nodes have sufficient RAM and Storage.")
    else:
        print("One or more nodes lack sufficient RAM or Storage.")
        # Optionally, print details of insufficient nodes
        insufficient_nodes = node_resource_comparison[~(node_resource_comparison['RAM_Sufficient'] & node_resource_comparison['Storage_Sufficient'])]
        print("Details of insufficient nodes:")
        print(insufficient_nodes[['id', 'Available RAM', 'Required RAM', 'Available Storage', 'Required Storage', 'RAM_Sufficient', 'Storage_Sufficient']])

    # Return the comparison DataFrame
    return node_resource_comparison[['id', 'Available RAM', 'Required RAM', 'Available Storage', 'Required Storage', 'RAM_Sufficient', 'Storage_Sufficient']]

