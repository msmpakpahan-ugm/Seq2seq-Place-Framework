import pandas as pd
import json
import networkx as nx
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import sys # For float('inf') or large number


def json_to_dataframe(json_file_path):
    # Convert application JSON to DataFrame
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    rows = []
    for app in data:
        app_id = app['id']
        numberofmodule = app['numberofmodule']
        # Ensure module and message lists have the same length or handle appropriately
        min_len = min(len(app.get('module', [])), len(app.get('message', [])))
        for i in range(min_len):
            module = app['module'][i]
            message = app['message'][i]
            row = {
                'Application ID': app_id,
                'numberofmodule': numberofmodule,
                'Module ID': module.get('id'),
                'Module Name': module.get('name'),
                'Required RAM': module.get('RAM'), # Keep RAM if needed elsewhere, otherwise remove
                'Required Storage': module.get('Storage'), # Added Required Storage
                'Message ID': message.get('id'),
                'Message Name': message.get('name'),
                'Bytes': message.get('bytes'),
                'Instructions': message.get('instructions')
            }
            # Handle cases where Storage might be missing in the input JSON
            if row['Required Storage'] is None:
                print(f"Warning: Module {module.get('id')} in App {app_id} is missing 'Storage' field. Defaulting to 0.")
                row['Required Storage'] = 0
            rows.append(row)
    return pd.DataFrame(rows)

def extract_nodes_to_dataframe(filepath):
    # Convert topology JSON to DataFrame
    with open(filepath, 'r') as file:
        data = json.load(file)
    nodes_data = data['nodes']
    # Ensure 'Storage' column exists, add default if not
    for node in nodes_data:
        if 'Storage' not in node:
            print(f"Warning: Node {node.get('id')} is missing 'Storage' field. Defaulting to 0.")
            node['Storage'] = 0
    return pd.DataFrame(nodes_data)

def json_to_graph_using_links(filepath):
    # Convert topology JSON to Graph
    with open(filepath, 'r') as file:
        data = json.load(file)
    G = nx.Graph()
    for node in data['nodes']:
        G.add_node(node['id'])
    if 'links' in data:
        for link in data['links']:
            G.add_edge(link['source'], link['target'], weight=link.get('weight', 1))
    return G

def flatten_json_to_dataframe(json_file_path):
    # Convert population JSON to DataFrame
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return pd.json_normalize(data, sep='_')

def fully_flatten_population_data(json_file_path):
    # Fully flatten population JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return pd.json_normalize(data, record_path=['sources'], sep='_')

# --- Constants for Energy Calculation ---
TRANSMISSION_POWER = 9.5
NODE_PROCESSING_POWER = 65
CLOUD_PROCESSING_POWER = 200
# --- Fitness Weights (Adjust as needed) ---
# Weights determine the importance of each objective (higher weight = more important)
# Since we minimize, a higher value for an objective contributes more negatively to fitness.
W_RESPONSE_TIME = 0.4
W_COST = 0.3
W_ENERGY = 0.3
W_STORAGE_PENALTY = 1.0 # Weight for storage violation penalty
# --- Define a very large number for penalty cases ---
LARGE_PENALTY_VALUE = 1e12 # Or sys.float_info.max

class GeneticAlgorithmv5additionalfitness:
    def __init__(self, nodes_df, app_df, graph, population_df, population_size, elitism_rate, crossover_prob, mutation_rate, shortest_paths_file=None):
        self.shortest_paths_file = shortest_paths_file
        self.nodes_df = nodes_df
        self.app_df = app_df
        self.graph = graph
        self.population_df = population_df # Used for app source nodes
        self.population_size = population_size
        self.elitism_rate = elitism_rate
        self.crossover_prob = crossover_prob
        self.mutation_rate = mutation_rate

        # --- Precompute/Cache Data ---
        self.precompute_shortest_paths()
        self.initialize_applications_and_cache() # Combined initialization

        # --- NEW: Store processing powers ---
        self.node_processing_powers = {
            node_id: CLOUD_PROCESSING_POWER if node_id == 100 else NODE_PROCESSING_POWER
            for node_id in self.nodes_df['id']
        }
        # Ensure cloud (100) is included if it's in the nodes list
        if 100 not in self.node_processing_powers and 100 in self.nodes_df['id'].values:
             self.node_processing_powers[100] = CLOUD_PROCESSING_POWER

        # --- NEW: Track Global Min/Max for Historical Normalization ---
        self.global_min_max = {
            'response_time': {'min': float('inf'), 'max': -float('inf')},
            'cost': {'min': float('inf'), 'max': -float('inf')},
            'energy': {'min': float('inf'), 'max': -float('inf')}
        }

        # --- Initialize population ---
        self.population = self.initialize_population() # Keep random initialization for now


    def plot_fitness_progress(self, fitness_history):
        # Ensure fitness_history contains only numeric scores
        valid_fitness_history = [f for f in fitness_history if isinstance(f, (int, float))]
        if not valid_fitness_history:
            print("No valid fitness history to plot.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(valid_fitness_history) + 1), valid_fitness_history)
        plt.title('Fitness Value vs Number of Generations')
        plt.xlabel('Number of Generations')
        plt.ylabel('Fitness Value (Higher is Better)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()


    def precompute_shortest_paths(self):
        if self.shortest_paths_file and os.path.isfile(self.shortest_paths_file):
            self.shortest_paths_table = pd.read_csv(self.shortest_paths_file, index_col='From')
            # Ensure columns are treated as the same type as the index (node IDs)
            try:
                 col_type = self.shortest_paths_table.index.dtype
                 self.shortest_paths_table.columns = self.shortest_paths_table.columns.astype(col_type)
            except Exception as e:
                 print(f"Warning: Could not convert shortest path column types - {e}")
            print(f"Reading shortest paths from {self.shortest_paths_file}")
        else:
            print("Computing shortest paths...")
            shortest_paths_dict = dict(nx.all_pairs_shortest_path_length(self.graph))
            # Convert dict to DataFrame, fill missing paths with a large number or inf
            self.shortest_paths_table = pd.DataFrame(shortest_paths_dict).fillna(np.inf) # Use np.inf for missing paths
            self.shortest_paths_table.index.name = 'From'
            self.shortest_paths_table.columns.name = 'To'
            if self.shortest_paths_file:
                try:
                    self.shortest_paths_table.to_csv(self.shortest_paths_file)
                    print(f"Saved shortest paths to {self.shortest_paths_file}")
                except Exception as e:
                    print(f"Error saving shortest paths: {e}")
        # Replace np.inf with our large penalty value for easier arithmetic later? Or handle inf directly.
        # Handling inf directly is usually better.
        # print(self.shortest_paths_table.head())


    def initialize_applications_and_cache(self):
        # Cache application source nodes
        try:
            # Assuming 'app' is string in population_df and integer in app_df
            self.app_to_source_node = {int(k): v for k, v in self.population_df.set_index('app')['id_resource'].to_dict().items()}
        except KeyError:
             print("Error: 'app' or 'id_resource' column not found in population_df. Cannot map app sources.")
             self.app_to_source_node = {}
        except Exception as e:
            print(f"Error initializing app_to_source_node: {e}")
            self.app_to_source_node = {}


        # Cache module Storage requirements
        if 'Required Storage' not in self.app_df.columns:
            raise ValueError("Error: 'Required Storage' column not found in app_df. Cannot proceed with Storage validation.")
        self.module_to_storage = self.app_df.set_index(['Application ID', 'Module ID'])['Required Storage'].to_dict()

        # --- NEW: Cache module message bytes and instructions ---
        # Assuming 'Bytes' and 'Instructions' are associated with the message *output* by the module
        self.module_bytes = self.app_df.set_index(['Application ID', 'Module ID'])['Bytes'].to_dict()
        self.module_instructions = self.app_df.set_index(['Application ID', 'Module ID'])['Instructions'].to_dict()

        # --- NEW: Cache node IPT, Cost, and Storage ---
        self.node_ipts = self.nodes_df.set_index('id')['IPT'].to_dict()
        # Handle potential missing 'Cost' column gracefully
        if 'Cost' in self.nodes_df.columns:
             self.node_costs = self.nodes_df.set_index('id')['Cost'].to_dict()
        else:
             print("Warning: 'Cost' column not found in nodes_df. Cost objective will be zero.")
             self.node_costs = {node_id: 0 for node_id in self.nodes_df['id']} # Default cost to 0

        # Cache node Storage
        if 'Storage' not in self.nodes_df.columns:
            raise ValueError("Error: 'Storage' column not found in nodes_df. Cannot proceed with Storage validation.")
        self.node_storage = self.nodes_df.set_index('id')['Storage'].to_dict()


        # Cache app modules structure
        self.app_modules = {app_id: sorted(self.app_df[self.app_df['Application ID'] == app_id]['Module ID'].unique())
                          for app_id in self.app_df['Application ID'].unique()}


    def initialize_population(self):
        population = []
        node_ids = list(self.nodes_df['id']) # Get all possible node IDs
        if not node_ids:
             raise ValueError("No nodes available to initialize population.")

        # Use module_to_storage keys now
        all_modules = list(self.module_to_storage.keys()) # Get all (App ID, Module ID) tuples

        for _ in range(self.population_size):
            chromosome = {}
            for module_key in all_modules:
                # Randomly assign a node ID to each module
                chromosome[module_key] = random.choice(node_ids)
            population.append(chromosome)
        return population

    # --- chromosome_to_dataframe remains the same ---
    # (Make sure they are present in your class)
    def chromosome_to_dataframe(self, chromosome, app_df):
        # Convert the chromosome dictionary to a DataFrame
        # Ensure chromosome keys are tuples (AppID, ModuleID)
        valid_items = [(k, v) for k, v in chromosome.items() if isinstance(k, tuple) and len(k) == 2]
        if not valid_items: return pd.DataFrame(columns=['Module Name', 'Application ID', 'Node ID']) # Return empty if no valid items

        chromosome_df = pd.DataFrame(valid_items, columns=['ModuleKey', 'Node ID'])
        chromosome_df[['Application ID', 'Module ID']] = pd.DataFrame(chromosome_df['ModuleKey'].tolist(), index=chromosome_df.index)

        # Merge with the application DataFrame to get the Module Name
        # Ensure correct dtypes for merging if necessary
        app_df_subset = app_df[['Application ID', 'Module ID', 'Module Name']].drop_duplicates()
        try:
             merged_df = pd.merge(chromosome_df, app_df_subset, on=['Application ID', 'Module ID'], how='left')
        except Exception as e:
             print(f"Error merging chromosome with app_df: {e}")
             # Handle potential type mismatches, e.g., ensure IDs are same type
             app_df_subset['Application ID'] = app_df_subset['Application ID'].astype(chromosome_df['Application ID'].dtype)
             app_df_subset['Module ID'] = app_df_subset['Module ID'].astype(chromosome_df['Module ID'].dtype)
             merged_df = pd.merge(chromosome_df, app_df_subset, on=['Application ID', 'Module ID'], how='left')


        # Rearrange columns and drop the intermediate 'ModuleKey' column
        placement_df = merged_df[['Module Name', 'Application ID', 'Node ID']].dropna() # Drop rows where merge failed


        return placement_df

    def check_placement_validity(self, nodes_df, application_df, placement_df):
        # Checks if Storage constraints are met.
        # Returns 0 if valid, or the total positive storage overflow if invalid.
        if placement_df.empty:
            return 0 # No placements means no violations

        # Merge placement with application data to get Storage requirements
        app_storage_df = application_df[['Application ID', 'Module Name', 'Required Storage']].drop_duplicates(subset=['Application ID', 'Module Name'])
        try:
            placement_app_merged = pd.merge(placement_df, app_storage_df, on=['Module Name', 'Application ID'], how='left')
        except Exception as e:
            print(f"Error merging placement and app Storage data: {e}")
            try:
                app_storage_df['Application ID'] = app_storage_df['Application ID'].astype(placement_df['Application ID'].dtype)
                placement_app_merged = pd.merge(placement_df, app_storage_df, on=['Module Name', 'Application ID'], how='left')
            except Exception as e2:
                print(f"Retry merge failed: {e2}")
                # Return a large overflow if validity cannot be checked? Or handle differently.
                return LARGE_PENALTY_VALUE # Indicate failure to check

        # Check for NaNs in 'Required Storage' after merge
        if placement_app_merged['Required Storage'].isnull().any():
            print("Warning: Some modules in placement_df missing Storage info after merge. These contribute to potential overflow calculation errors.")
            # Option 1: Treat modules with missing info as having 0 requirement (might be too lenient)
            # placement_app_merged['Required Storage'].fillna(0, inplace=True)
            # Option 2: Exclude them (as currently done) - might underestimate overflow if these *should* be placed
            # Option 3: Return a large penalty value if any info is missing?
            placement_app_merged.dropna(subset=['Required Storage'], inplace=True) # Keep current approach for now

        # Calculate Storage usage per node
        node_storage_usage = placement_app_merged.groupby('Node ID')['Required Storage'].sum().reset_index()

        # Get available Storage from nodes_df
        if 'Storage' not in nodes_df.columns:
            print("Error: 'Storage' column missing in nodes_df. Cannot check validity.")
            return LARGE_PENALTY_VALUE # Indicate critical error

        nodes_available_storage = nodes_df[['id', 'Storage']].rename(columns={'Storage': 'Available Storage'})

        # Compare available and required Storage
        node_storage_comparison = pd.merge(nodes_available_storage, node_storage_usage, left_on='id', right_on='Node ID', how='left')
        node_storage_comparison['Required Storage'] = node_storage_comparison['Required Storage'].fillna(0)

        # Calculate overflow per node
        node_storage_comparison['Overflow'] = (node_storage_comparison['Required Storage'] - node_storage_comparison['Available Storage']).clip(lower=0) # Keep only positive differences

        # Calculate total overflow
        total_overflow = node_storage_comparison['Overflow'].sum()

        return total_overflow # Returns 0 if valid, > 0 if invalid

    def print_placement_validity(self, nodes_df, application_df, placement_df):
        # Similar to check_placement_validity but returns the comparison DataFrame based on Storage
        if placement_df.empty:
             print("Placement DataFrame is empty.")
             return pd.DataFrame(columns=['id', 'Available Storage', 'Required Storage', 'Overflow'])

        # Merge placement with application data to get Storage requirements
        app_storage_df = application_df[['Application ID', 'Module Name', 'Required Storage']].drop_duplicates(subset=['Application ID', 'Module Name'])
        try:
            placement_app_merged = pd.merge(placement_df, app_storage_df, on=['Module Name', 'Application ID'], how='left')
        except Exception as e:
             print(f"Error merging placement and app Storage data: {e}")
             return pd.DataFrame(columns=['id', 'Available Storage', 'Required Storage', 'Overflow'])

        if placement_app_merged['Required Storage'].isnull().any():
             print("Warning: Some modules in placement_df missing Storage info after merge.")
             placement_app_merged.dropna(subset=['Required Storage'], inplace=True)

        # Calculate Storage usage per node
        node_storage_usage = placement_app_merged.groupby('Node ID')['Required Storage'].sum().reset_index()

        # Get available Storage from nodes_df
        if 'Storage' not in nodes_df.columns:
              print("Error: 'Storage' column missing in nodes_df. Cannot print validity.")
              return pd.DataFrame(columns=['id', 'Available Storage', 'Required Storage', 'Overflow'])
        nodes_available_storage = nodes_df[['id', 'Storage']].rename(columns={'Storage': 'Available Storage'})

        # Compare available and required Storage
        node_storage_comparison = pd.merge(nodes_available_storage, node_storage_usage, left_on='id', right_on='Node ID', how='left')
        node_storage_comparison['Required Storage'] = node_storage_comparison['Required Storage'].fillna(0)
        node_storage_comparison['Overflow'] = (node_storage_comparison['Required Storage'] - node_storage_comparison['Available Storage']).clip(lower=0)
        node_storage_comparison['Storage_Sufficient'] = node_storage_comparison['Overflow'] <= 0 # Add sufficiency boolean back for clarity if needed

        # Return relevant columns
        return node_storage_comparison[['id', 'Available Storage', 'Required Storage', 'Overflow', 'Storage_Sufficient']]


    def fitness(self, chromosome):
        # --- Objective Initialization ---
        total_latency = 0
        total_processing_time = 0
        total_cost = 0
        total_energy = 0
        used_nodes = set() # Track unique nodes for cost calculation

        # --- Iterate through applications ---
        for app_id, module_ids in self.app_modules.items():
            source_node_id = self.app_to_source_node.get(app_id)
            if source_node_id is None:
                # print(f"Warning: No source node defined for app {app_id}. Skipping app.")
                continue # Skip app if source node isn't defined

            modules_in_app = [(app_id, mod_id) for mod_id in module_ids]

            # --- Calculate Processing Time and Track Used Nodes ---
            app_processing_time = 0
            nodes_for_this_app = set() # Track nodes used by this specific app's modules
            for module_key in modules_in_app: # module_key is (app_id, mod_id)
                node_id = chromosome.get(module_key)
                if node_id is None:
                    # print(f"Warning: Module {module_key} not found in chromosome. Assigning large penalty.")
                    app_processing_time += LARGE_PENALTY_VALUE # Penalize if module missing
                    continue

                used_nodes.add(node_id)
                nodes_for_this_app.add(node_id)

                instructions = self.module_instructions.get(module_key, 0)
                node_ipt = self.node_ipts.get(node_id, 0)

                if node_ipt > 1e-9: # Avoid division by zero or near-zero
                    processing_time_module = instructions / node_ipt
                else:
                    # print(f"Warning: Node {node_id} has IPT {node_ipt}. Assigning large processing time penalty.")
                    processing_time_module = LARGE_PENALTY_VALUE # Penalize if IPT is zero/too small

                app_processing_time += processing_time_module

                # Calculate processing energy for this module
                node_proc_power = self.node_processing_powers.get(node_id, NODE_PROCESSING_POWER) # Default to node power
                total_energy += processing_time_module * node_proc_power

            total_processing_time += app_processing_time


            # --- Calculate Latency (Communication Time) ---
            app_latency = 0
            if modules_in_app:
                # Communication from Source to First Module
                first_module_key = modules_in_app[0]
                first_node_id = chromosome.get(first_module_key)
                if first_node_id is not None:
                    try:
                        hops = self.shortest_paths_table.at[source_node_id, first_node_id]
                        if np.isinf(hops): hops = LARGE_PENALTY_VALUE # Penalize if no path

                        # Assuming message from source triggers first module, use first module's output msg size? Or define source message size?
                        # Let's use the first module's output message bytes for this hop. Adjust if needed.
                        message_bytes = self.module_bytes.get(first_module_key, 0)
                        # Latency = (Bytes / BW) * Hops. BW = 1 (assumed)
                        latency_link = message_bytes * hops
                        app_latency += latency_link
                        total_energy += latency_link * TRANSMISSION_POWER # Transmission energy

                    except KeyError:
                        # print(f"Warning: Path lookup failed between {source_node_id} and {first_node_id}. Assigning large latency penalty.")
                        app_latency += LARGE_PENALTY_VALUE
                        total_energy += LARGE_PENALTY_VALUE * TRANSMISSION_POWER


                # Communication Between Modules (Sequential)
                for i in range(len(modules_in_app) - 1):
                    from_module_key = modules_in_app[i]
                    to_module_key = modules_in_app[i+1]

                    from_node_id = chromosome.get(from_module_key)
                    to_node_id = chromosome.get(to_module_key)

                    if from_node_id is not None and to_node_id is not None and from_node_id != to_node_id: # Only if different nodes
                        try:
                            hops = self.shortest_paths_table.at[from_node_id, to_node_id]
                            if np.isinf(hops): hops = LARGE_PENALTY_VALUE # Penalize if no path

                            # Message size is from the sending module (from_module_key)
                            message_bytes = self.module_bytes.get(from_module_key, 0)
                            latency_link = message_bytes * hops # BW = 1 assumed
                            app_latency += latency_link
                            total_energy += latency_link * TRANSMISSION_POWER # Transmission energy

                        except KeyError:
                            # print(f"Warning: Path lookup failed between {from_node_id} and {to_node_id}. Assigning large latency penalty.")
                            app_latency += LARGE_PENALTY_VALUE
                            total_energy += LARGE_PENALTY_VALUE * TRANSMISSION_POWER

            total_latency += app_latency


        # --- Calculate Total Cost ---
        # Sum cost of unique nodes used by any module
        total_cost = sum(self.node_costs.get(node_id, 0) for node_id in used_nodes)


        # --- Calculate Total Response Time ---
        total_response_time = total_latency + total_processing_time


        # --- Check Placement Validity (Uses Storage now) ---
        placement_df = self.chromosome_to_dataframe(chromosome, self.app_df)
        storage_violation = self.check_placement_validity(self.nodes_df, self.app_df, placement_df)


        # --- Store Individual Objective Values ---
        # Return raw components and validity; normalization and scoring happen in 'run'
        components = {
            'total_response_time': total_response_time,
            'total_cost': total_cost,
            'total_energy': total_energy,
            'storage_violation_amount': storage_violation # Store violation amount (0 if valid)
        }

        # Return raw components, final fitness calculated in run method after normalization and penalty
        return components


    # --- select, crossover, mutate ---
    # These methods likely don't need changes as they operate on the chromosome structure.
    # Ensure self.select uses the fitness score correctly (first element of the tuple).
    def select(self, population_fitness_tuples):
        # Tournament selection based on fitness score
        if not population_fitness_tuples:
             raise ValueError("Selection pool is empty")

        tournament_size = min(5, len(population_fitness_tuples))
        tournament_individuals = random.sample(population_fitness_tuples, tournament_size)

        # Find the best individual (chromosome, (score, components)) in the tournament based on score
        best_individual = max(tournament_individuals, key=lambda item: item[1][0]) # item[1][0] is the fitness score

        return best_individual[0] # Return the chromosome of the winner


    def crossover(self, parent1_chromo, parent2_chromo):
         # Single-point crossover (example)
        if random.random() < self.crossover_prob:
            keys = list(parent1_chromo.keys())
            if len(keys) < 2: return parent1_chromo.copy() # Cannot perform crossover

            crossover_point = random.randint(1, len(keys) - 1)
            child_chromo = {}
            for i, key in enumerate(keys):
                if i < crossover_point:
                    child_chromo[key] = parent1_chromo[key]
                else:
                    # Ensure the key exists in parent2 before accessing
                    child_chromo[key] = parent2_chromo.get(key, parent1_chromo[key]) # Fallback to parent1 if key missing in parent2
            return child_chromo
        else:
            # Return one of the parents if no crossover
            return parent1_chromo.copy() if random.random() < 0.5 else parent2_chromo.copy()


    def mutate(self, chromosome):
        # Randomly change the node assignment for some modules
        node_ids = list(self.nodes_df['id'])
        if not node_ids: return chromosome # Cannot mutate if no nodes

        mutated_chromosome = chromosome.copy()
        for module_key in mutated_chromosome:
            if random.random() < self.mutation_rate:
                mutated_chromosome[module_key] = random.choice(node_ids)
        return mutated_chromosome


    # --- run Method ---
    def run(self, num_iterations):
        best_fitness_overall = -float('inf')
        best_chromosome_overall = None
        best_components_overall = None
        fitness_history = []

        for iteration in range(num_iterations):
            # 1. Calculate raw components for the entire population
            population_components = [(chromo, self.fitness(chromo)) for chromo in self.population]

            # 2. Identify valid individuals and collect objective values for normalization range
            valid_individuals = []
            all_violations = []

            # --- Adaptive Max Reset (after 30 generations) ---
            if iteration >= 30:
                # Reset max values to find the max of the current generation
                self.global_min_max['response_time']['max'] = -float('inf')
                self.global_min_max['cost']['max'] = -float('inf')
                self.global_min_max['energy']['max'] = -float('inf')
                # Note: Min values are NOT reset, they track the true historical best

            for chromo, components in population_components:
                violation = components['storage_violation_amount']
                all_violations.append(violation)
                
                # --- Update Global Min/Max using ALL individuals --- 
                rt = components['total_response_time']
                cost = components['total_cost']
                energy = components['total_energy']
                # Min always tracks the historical best across all individuals seen
                self.global_min_max['response_time']['min'] = min(self.global_min_max['response_time']['min'], rt)
                self.global_min_max['cost']['min'] = min(self.global_min_max['cost']['min'], cost)
                self.global_min_max['energy']['min'] = min(self.global_min_max['energy']['min'], energy)
                # Max tracks historical max for first 30 gens, then current gen max (based on all individuals)
                self.global_min_max['response_time']['max'] = max(self.global_min_max['response_time']['max'], rt)
                self.global_min_max['cost']['max'] = max(self.global_min_max['cost']['max'], cost)
                self.global_min_max['energy']['max'] = max(self.global_min_max['energy']['max'], energy)

                # Identify valid individuals (still needed for potential analysis/logging later, though not for min/max tracking)
                if violation < 1e-9:
                    valid_individuals.append((chromo, components))
                    
            # 3. Determine normalization ranges based on potentially adaptive GLOBAL values (updated using all individuals)
            rt_range = self.global_min_max['response_time']['max'] - self.global_min_max['response_time']['min']
            cost_range = self.global_min_max['cost']['max'] - self.global_min_max['cost']['min']
            energy_range = self.global_min_max['energy']['max'] - self.global_min_max['energy']['min']

            # Avoid division by zero
            rt_range = rt_range if rt_range > 1e-9 else 1.0
            cost_range = cost_range if cost_range > 1e-9 else 1.0
            energy_range = energy_range if energy_range > 1e-9 else 1.0

            # 5. Calculate final fitness for ALL individuals
            population_fitness_results = []
            for chromo, components in population_components:
                # Normalize objectives using potentially adaptive GLOBAL ranges (now based on all individuals)
                norm_rt = (components['total_response_time'] - self.global_min_max['response_time']['min']) / rt_range
                norm_cost = (components['total_cost'] - self.global_min_max['cost']['min']) / cost_range
                norm_energy = (components['total_energy'] - self.global_min_max['energy']['min']) / energy_range

                # --- Calculate fitness based on normalized objectives --- 
                obj_denominator = norm_rt + norm_cost + norm_energy
                if obj_denominator < 1e-9:
                    base_fitness_part = float('inf') 
                else:
                    base_fitness_part = 3.0 / obj_denominator
                    
                # --- Get raw violation --- 
                violation = components['storage_violation_amount']
                
                # --- Apply penalty by subtracting weighted RAW violation --- 
                final_fitness = base_fitness_part - W_STORAGE_PENALTY * violation
                
                # Handle edge case: inf - penalty
                if violation > 1e-9 and base_fitness_part == float('inf'):
                     final_fitness = (1.0 / 1e-9) - W_STORAGE_PENALTY * violation # Use a large defined value

                # --- Store normalized values in components dict for logging --- 
                components['norm_response_time'] = norm_rt
                components['norm_cost'] = norm_cost
                components['norm_energy'] = norm_energy
                
                # Store results: chromosome, (final_fitness, components_with_norm_values)
                population_fitness_results.append((chromo, (final_fitness, components)))

            # --- Proceed with GA steps using calculated final fitness scores ---

            if not population_fitness_results:
                print(f"Warning: Population empty after fitness calculation for iteration {iteration}. Stopping.")
                break

            # Sort population based on final_fitness (higher is better)
            sorted_population_tuples = sorted(population_fitness_results,
                                            key=lambda x: x[1][0], # Sort by final_fitness
                                            reverse=True)

            # Get best of current generation
            current_best_chromosome = sorted_population_tuples[0][0]
            current_best_fitness, current_best_components = sorted_population_tuples[0][1]
            # Add final fitness score to components dict for logging/return
            current_best_components['final_fitness_score'] = current_best_fitness

            # Track overall best (based on final fitness score)
            if current_best_fitness > best_fitness_overall:
                best_fitness_overall = current_best_fitness
                best_chromosome_overall = current_best_chromosome
                best_components_overall = current_best_components.copy() # Store a copy

            # Store fitness for plotting
            fitness_history.append(current_best_fitness)

            # --- Elitism ---
            num_elites = int(self.elitism_rate * self.population_size)
            elites = [item[0] for item in sorted_population_tuples[:num_elites]]

            # --- Build New Population ---
            new_population = elites[:] # Start with elites

            # Fill the rest using selection, crossover, mutation
            while len(new_population) < self.population_size:
                parent1_chromo = self.select(sorted_population_tuples)
                parent2_chromo = self.select(sorted_population_tuples)

                child_chromo = self.crossover(parent1_chromo, parent2_chromo)
                child_chromo = self.mutate(child_chromo)
                new_population.append(child_chromo)

            self.population = new_population # Update population for next iteration

            # --- Print Progress ---
            print(f'Iteration {iteration}: Best Fitness = {current_best_fitness:.4f}, Overall Best Fitness = {best_fitness_overall:.4f}')
            print("  Current Best Components (Raw & Normalized):")
            temp_best_components = sorted_population_tuples[0][1][1] # Get components dict
            # Define keys for raw and normalized values for cleaner printing logic
            raw_keys = ['total_response_time', 'total_cost', 'total_energy', 'storage_violation_amount']
            norm_keys = ['norm_response_time', 'norm_cost', 'norm_energy']
            
            print("    Raw Values:")
            for key in raw_keys:
                if key in temp_best_components:
                    value = temp_best_components[key]
                    print(f"      {key}: {value:.4f}")
                    
            print("    Normalized Values:")
            for key in norm_keys:
                 if key in temp_best_components:
                    value = temp_best_components[key]
                    print(f"      {key}: {value:.4f}")

        # --- Final Results ---
        print("\nGenetic Algorithm Complete!")
        print("Final Best Solution:")
        if best_chromosome_overall and best_components_overall:
            print(f"  Overall Best Fitness (Calculated as [3 / norm_objs] - W_penalty * raw_violation): {best_fitness_overall:.4f}")
            print("  Best Solution Components (Raw & Normalized):")
            # Reuse printing logic
            print("    Raw Values:")
            for key in raw_keys:
                 if key in best_components_overall:
                    value = best_components_overall[key]
                    print(f"      {key}: {value:.4f}")
            print("    Normalized Values:")
            for key in norm_keys:
                 if key in best_components_overall:
                    value = best_components_overall[key]
                    print(f"      {key}: {value:.4f}")

            print("  Placement:")
            sorted_placement = sorted(best_chromosome_overall.items())
            for module_key, node_id in sorted_placement:
                print(f"    App {module_key[0]}, Mod {module_key[1]} -> Node {node_id}")
        else:
            print("  No valid solution found or population empty.")

        # --- Plotting ---
        self.plot_fitness_progress(fitness_history)

        # Return the best solution found
        return best_chromosome_overall, best_fitness_overall, best_components_overall

    # --- output_placement_json, trim_and_output_placement_json ---
    # These methods should still work if they rely on the chromosome structure.
    # Make sure they are included in your class definition.
    # (Copying trim_and_output_placement_json from previous context for completeness)
    def trim_and_output_placement_json(self, chromosome, nodes_df, application_df, output_file_path):
        # Convert chromosome to DataFrame
        # --- Reuse chromosome_to_dataframe logic ---
        placement_df = self.chromosome_to_dataframe(chromosome, application_df)
        if placement_df.empty:
            print("Cannot generate output JSON: Chromosome resulted in empty placement.")
            # Decide what to do: return error, empty file, etc.
            # Example: Write empty allocation
            converted_data = {"initialAllocation": []}
            try:
                with open(output_file_path, 'w') as file:
                    json.dump(converted_data, file, indent=4)
                return 0 # Return 0 failed apps
            except Exception as e:
                return f"Error writing empty JSON file: {e}"


        # Merge with the application DataFrame to get Storage requirements
        app_storage_df = application_df[['Application ID', 'Module Name', 'Required Storage']].drop_duplicates(subset=['Application ID', 'Module Name'])
        try:
            merged_df = pd.merge(placement_df, app_storage_df, on=['Module Name', 'Application ID'], how='left')
        except Exception as e:
             print(f"Error merging placement and app Storage data for output: {e}")
             # Handle potential type mismatches
             try:
                 app_storage_df['Application ID'] = app_storage_df['Application ID'].astype(placement_df['Application ID'].dtype)
                 merged_df = pd.merge(placement_df, app_storage_df, on=['Module Name', 'Application ID'], how='left')
             except Exception as e2:
                  print(f"Retry merge failed for output: {e2}")
                  # Cannot proceed without Storage info
                  return "Error: Could not merge placement with Storage data for output JSON."


        # Ensure 'Required Storage' column exists and handle NaNs
        if 'Required Storage' not in merged_df.columns:
             print("Error: 'Required Storage' column missing after merge for output.")
             return "Error: Missing Storage data for output JSON."
        if merged_df['Required Storage'].isnull().any():
             print("Warning: Some modules missing Storage info in merged output data. These cannot be placed.")
             # Keep only rows with valid Storage info
             merged_df.dropna(subset=['Required Storage'], inplace=True)


        # Make a copy of nodes_df to track available Storage during trimming
        # Need 'id' and 'Storage' columns
        if 'id' not in nodes_df.columns or 'Storage' not in nodes_df.columns:
             return "Error: nodes_df must contain 'id' and 'Storage' columns for trimming."
        nodes_storage_available = nodes_df[['id', 'Storage']].copy()
        nodes_storage_available.rename(columns={'Storage': 'Available Storage'}, inplace=True)
        nodes_storage_available.set_index('id', inplace=True)


        valid_placements = []
        failed_modules_count = 0
        failed_apps_set = set()


        # Iterate over each potential placement
        for index, row in merged_df.iterrows():
            node_id = row['Node ID']
            required_storage = row['Required Storage'] # Changed from required_ram
            app_id = row['Application ID']
            # Need Module ID for the output format "Mod{module_id}"
            # Get Module ID by merging back or ensuring it's carried through
            module_id = None # Placeholder - Need to retrieve Module ID
            app_subset = application_df[(application_df['Application ID'] == app_id) & (application_df['Module Name'] == row['Module Name'])]
            if not app_subset.empty:
                 module_id = app_subset['Module ID'].iloc[0]
            else:
                 print(f"Warning: Could not find Module ID for App {app_id}, Module Name {row['Module Name']}")
                 failed_modules_count += 1
                 failed_apps_set.add(app_id)
                 continue # Skip if module ID cannot be found


            # Check if node exists in our Storage tracking
            if node_id not in nodes_storage_available.index:
                 print(f"Warning: Node ID {node_id} from placement not found in nodes_df for Storage check. Skipping module.")
                 failed_modules_count += 1
                 failed_apps_set.add(app_id)
                 continue


            # Check if the node has enough available Storage
            if nodes_storage_available.at[node_id, 'Available Storage'] >= required_storage:
                # Deduct the required Storage
                nodes_storage_available.at[node_id, 'Available Storage'] -= required_storage


                # Add to valid placements list for JSON output
                valid_placements.append({
                    "module_name": f"Mod{module_id}", # Use the retrieved Module ID
                    "app": str(app_id),
                    "id_resource": int(node_id) # Ensure node ID is integer
                })
            else:
                # Increment failed counters if Storage is insufficient
                failed_modules_count += 1
                failed_apps_set.add(app_id)


        print(f"Trimming complete. Number of module placements failed due to Storage constraints: {failed_modules_count}")
        print(f"Number of applications with at least one failed placement: {len(failed_apps_set)}")


        # Format for the specific JSON structure
        converted_data = {"initialAllocation": valid_placements}


        # Write the valid placements to the JSON file
        try:
             # Check if file exists - overwrite or error based on desired behavior
             # Original code returned error if exists. Let's overwrite instead.
             # if os.path.exists(output_file_path):
             #    return f"Error: File already exists at {output_file_path}."

             with open(output_file_path, 'w') as file:
                 json.dump(converted_data, file, indent=4)
             print(f"Successfully wrote trimmed placement to {output_file_path}")

        except Exception as e:
             return f"Error writing output JSON file: {e}"


        # Return the number of applications that had failures
        return len(failed_apps_set)