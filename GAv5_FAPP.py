import pandas as pd
import json
import networkx as nx

def json_to_dataframe(json_file_path):
    # Convert application JSON to DataFrame
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    rows = []
    for app in data:
        app_id = app['id']
        numberofmodule = app['numberofmodule']
        for module, message in zip(app['module'], app['message']):
            row = {
                'Application ID': app_id,
                'numberofmodule': numberofmodule,
                'Module ID': module['id'],
                'Module Name': module['name'],
                'Required RAM': module['RAM'],
                'Message ID': message['id'],
                'Message Name': message['name'],
                'Bytes': message['bytes'],
                'Instructions': message['instructions']
            }
            rows.append(row)
    return pd.DataFrame(rows)

def extract_nodes_to_dataframe(filepath):
    # Convert topology JSON to DataFrame
    with open(filepath, 'r') as file:
        data = json.load(file)
    nodes_data = data['nodes']
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


# Before running the genetic algorithm, we need to define the GeneticAlgorithm class as discussed earlier
# with the proper implementation of the initialization, fitness function, selection, crossover, and mutation methods.
import pandas as pd
import json
import networkx as nx
import random
import numpy as np
import os
import matplotlib.pyplot as plt



class GeneticAlgorithmv5additionalfitness:
    def __init__(self, nodes_df, app_df, graph, population_df, population_size, elitism_rate, crossover_prob, mutation_rate, shortest_paths_file=None):
        self.shortest_paths_file = shortest_paths_file
        self.nodes_df = nodes_df
        self.app_df = app_df
        self.graph = graph
        self.population_df = population_df
        self.population_size = population_size
        self.population = self.initialize_population()
        self.elitism_rate = elitism_rate
        self.crossover_prob = crossover_prob
        self.mutation_rate = mutation_rate
        self.precompute_shortest_paths()
        self.initialize_applications()
        
        # Add these new cached properties
        self.node_ids = list(self.nodes_df['id'])
        self.app_modules = {app_id: sorted([module for module in self.app_df[self.app_df['Application ID'] == app_id]['Module ID']]) 
                          for app_id in self.app_df['Application ID'].unique()}
        self.module_bytes = self.app_df.set_index(['Application ID', 'Module ID'])['Bytes'].to_dict()
        
        # Pre-calculate max values for fitness normalization
        self.max_hops = self.nodes_df.shape[0] * len(self.app_df)
        self.max_message_size = self.app_df['Bytes'].sum() * self.nodes_df.shape[0]
        self.max_ipt = self.nodes_df[self.nodes_df['id'] != 100]['IPT'].max() * len(self.app_df)

    def plot_fitness_progress(self, fitness_history):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(fitness_history) + 1), fitness_history)
        plt.title('Fitness Value vs Number of Generations')
        plt.xlabel('Number of Generations')
        plt.ylabel('Fitness Value')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()


    def precompute_shortest_paths(self):
        if self.shortest_paths_file and os.path.isfile(self.shortest_paths_file):
            # Load the DataFrame from CSV
            self.shortest_paths_table = pd.read_csv(self.shortest_paths_file, index_col='From')
            self.shortest_paths_table.columns = self.shortest_paths_table.columns.astype(type(self.shortest_paths_table.index[0]))

            print(f"reading from {self.shortest_paths_file}")
        else:
            # Compute the shortest paths
            shortest_paths_dict = dict(nx.all_pairs_shortest_path_length(self.graph))
            self.shortest_paths_table = pd.DataFrame(shortest_paths_dict).fillna("N/A")
            self.shortest_paths_table.index.name = 'From'
            self.shortest_paths_table.columns.name = 'To'

            # Save the DataFrame to CSV if a file path is provided
            if self.shortest_paths_file:
                self.shortest_paths_table.to_csv(self.shortest_paths_file)
                print(f"saving to csv {self.shortest_paths_file}")

        # Print the table for viewing
        # print(self.shortest_paths_table)

    def initialize_applications(self):
        # Creating dictionaries for quick lookups instead of DataFrame access.
        self.app_to_source_node = self.population_df.set_index('app')['id_resource'].to_dict()
        self.module_to_ram = self.app_df.set_index(['Application ID', 'Module ID'])['Required RAM'].to_dict()


    ## not checking availibility but rather only listing the node id value.
    def initialize_population(self):
        # Initialize population with random placements
        population = []
        for _ in range(self.population_size):
            chromosome = {}
            for _, app in self.app_df.iterrows():
                module_id = (app['Application ID'], app['Module ID'])
                # Get all node IDs without checking RAM constraints
                possible_nodes = list(self.nodes_df['id'])
                chromosome[module_id] = random.choice(possible_nodes)
            population.append(chromosome)
        return population


    def chromosome_to_dataframe(self, chromosome, app_df):
        # Convert the chromosome dictionary to a DataFrame
        chromosome_df = pd.DataFrame(list(chromosome.items()), columns=['Module', 'Node ID'])
        chromosome_df[['Application ID', 'Module ID']] = pd.DataFrame(chromosome_df['Module'].tolist(), index=chromosome_df.index)

        # Merge with the application DataFrame to get the Module Name
        merged_df = pd.merge(chromosome_df, app_df[['Application ID', 'Module ID', 'Module Name']], on=['Application ID', 'Module ID'])

        # Rearrange columns and drop the 'Module' column
        placement_df = merged_df[['Module Name', 'Application ID', 'Node ID']]

        return placement_df

    # Main function to check placement validity
    def check_placement_validity(self, nodes_df, application_df, placement_df):
        # Merge and calculate RAM usage
        placement_app_merged = pd.merge(placement_df, application_df, on=['Module Name', 'Application ID'], how='left')
        node_ram_usage = placement_app_merged.groupby('Node ID')['Required RAM'].sum().reset_index()
        nodes_df = nodes_df.rename(columns={'RAM': 'Available RAM'})

        # Compare available and required RAM
        node_ram_comparison = pd.merge(nodes_df, node_ram_usage, left_on='id', right_on='Node ID', how='left')
        node_ram_comparison['Required RAM'].fillna(0, inplace=True)
        node_ram_comparison['RAM_Sufficient'] = node_ram_comparison['Available RAM'] >= node_ram_comparison['Required RAM']

        # Calculate the percentage of nodes with sufficient RAM
        total_nodes = len(node_ram_comparison)
        sufficient_nodes = node_ram_comparison['RAM_Sufficient'].sum()
        validity_percentage = (sufficient_nodes / total_nodes) * 100

        return validity_percentage

    def check_placement_validity(self, nodes_df, application_df, placement_df):
        # Merge and calculate RAM usage
        placement_app_merged = pd.merge(placement_df, application_df, on=['Module Name', 'Application ID'], how='left')
        node_ram_usage = placement_app_merged.groupby('Node ID')['Required RAM'].sum().reset_index()
        
        # Create a copy of nodes_df before modifying
        nodes_df = nodes_df.copy()
        nodes_df = nodes_df.rename(columns={'RAM': 'Available RAM'})
            # Compare available and required RAM
        node_ram_comparison = pd.merge(nodes_df, node_ram_usage, left_on='id', right_on='Node ID', how='left')
        # Replace fillna with assignment
        node_ram_comparison['Required RAM'] = node_ram_comparison['Required RAM'].fillna(0)
        node_ram_comparison['RAM_Sufficient'] = node_ram_comparison['Available RAM'] >= node_ram_comparison['Required RAM']
        return node_ram_comparison['RAM_Sufficient'].all()
    def print_placement_validity(self, nodes_df, application_df, placement_df):
    # Merge and calculate RAM usage
        placement_app_merged = pd.merge(placement_df, application_df, on=['Module Name', 'Application ID'], how='left')
        node_ram_usage = placement_app_merged.groupby('Node ID')['Required RAM'].sum().reset_index()
        
        # Create a copy of nodes_df before modifying
        nodes_df = nodes_df.copy()
        nodes_df = nodes_df.rename(columns={'RAM': 'Available RAM'})
            # Compare available and required RAM
        node_ram_comparison = pd.merge(nodes_df, node_ram_usage, left_on='id', right_on='Node ID', how='left')
        # Replace fillna with assignment
        node_ram_comparison['Required RAM'] = node_ram_comparison['Required RAM'].fillna(0)
        node_ram_comparison['RAM_Sufficient'] = node_ram_comparison['Available RAM'] >= node_ram_comparison['Required RAM']
        return node_ram_comparison[['id', 'Available RAM', 'Required RAM', 'RAM_Sufficient']]

    def fitness(self, chromosome):
        chromosome_key = tuple(sorted(chromosome.items()))  # Sort to ensure consistent caching
        if chromosome_key in self.fitness_cache:
            return self.fitness_cache[chromosome_key]

        total_hops = 0
        total_hops_message_size = 0
        used_fog_nodes = set()
        cloud_placements = 0
        num_modules = len(chromosome)

        # Use cached app_modules instead of recreating the list each time
        for app_id in self.app_modules:
            app_total_hops = 0
            app_total_hops_message_size = 0
            modules = [(app_id, module_id) for module_id in self.app_modules[app_id]]
            
            if modules:
                source_node_id = self.app_to_source_node[str(app_id)]
                prev_node = source_node_id
                
                # Use cached shortest paths and bytes
                first_module = modules[0]
                app_total_hops += self.shortest_paths_table.at[source_node_id, chromosome[first_module]]
                app_total_hops_message_size += (app_total_hops * 
                    self.module_bytes[(app_id, first_module[1])])

                for i in range(1, len(modules)):
                    curr_node = chromosome[modules[i]]
                    prev_node = chromosome[modules[i - 1]]
                    hops = self.shortest_paths_table.at[prev_node, curr_node]
                    app_total_hops += hops
                    app_total_hops_message_size += (hops * 
                        self.module_bytes[(app_id, modules[i][1])])
                    
                    used_fog_nodes.add(curr_node)
                    if curr_node == 100:
                        cloud_placements += 1

            total_hops += app_total_hops
            total_hops_message_size += app_total_hops_message_size

        # Use pre-calculated max values
        normalized_hops = total_hops / self.max_hops
        normalized_message_size = total_hops_message_size / self.max_message_size
        
        # Calculate IPT only for used fog nodes
        total_ipt = sum(self.nodes_df.loc[
            (self.nodes_df['id'].isin(used_fog_nodes)) & 
            (self.nodes_df['id'] != 100), 'IPT'])
        normalized_ipt = total_ipt / self.max_ipt

        # Calculate fitness score (higher is better)
        hop_weight = 0.3
        message_size_weight = 0.3
        ipt_weight = 0.4  # Increased weight for IPT

        fitness_score = (
            -hop_weight * normalized_hops
            - message_size_weight * normalized_message_size
            + ipt_weight * normalized_ipt  # Now we add IPT because we want to maximize it
        )

        # Placement validity percentage
        placement_validity_percentage = self.check_placement_validity(self.nodes_df, self.app_df, self.chromosome_to_dataframe(chromosome, self.app_df))
        placement_validity_bonus = (placement_validity_percentage / 100) * num_modules
        fitness_score += placement_validity_bonus

        # Penalty for cloud placements
        fitness_score -= cloud_placements

        components = {
            'total_hops': total_hops,
            'total_hops_message_size': total_hops_message_size,
            'total_ipt': total_ipt,
            'normalized_hops': normalized_hops,
            'normalized_message_size': normalized_message_size,
            'normalized_ipt': normalized_ipt,
            'placement_validity_percentage': placement_validity_percentage,
            'cloud_placements': cloud_placements
        }

        self.fitness_cache[chromosome_key] = (fitness_score, components)
        return fitness_score, components

    def select(self, pool):
        if not pool:
            raise ValueError("Selection pool is empty")
        tournament_size = min(5, len(pool))  # Use the smaller of 5 or the pool size
        tournament = random.sample(pool, tournament_size)
        return max(tournament, key=self.fitness)

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_prob:
            # Perform single-point crossover
            crossover_point = random.randint(1, len(parent1) - 1)
            child = {}
            for i, (service, node) in enumerate(parent1.items()):
                if i < crossover_point:
                    child[service] = parent1[service]
                else:
                    child[service] = parent2[service]
            return child
        else:
            # No crossover, return one of the parents
            return parent1.copy() if random.random() < 0.5 else parent2.copy()

    # also fixed to not check the possible nodes.
    def mutate(self, chromosome):
        for service_to_mutate in chromosome:
            if random.random() < self.mutation_rate:
                # Get all node IDs without checking RAM constraints
                possible_nodes = list(self.nodes_df['id'])
                chromosome[service_to_mutate] = random.choice(possible_nodes)
        return chromosome

    def run(self, num_iterations):
        # Initialize a cache for fitness values
        self.fitness_cache = {}
        print("Fitness cache initialized.")

        best_fitness_overall = float('-inf')
        best_chromosome_overall = None
        fitness_history = []

        # Batch process fitness calculations
        def calculate_population_fitness(population):
            return [(chromosome, self.fitness(chromosome)[0]) for chromosome in population]
        
        for iteration in range(num_iterations):
            new_population = []
            # Calculate fitness for entire population at once
            population_fitness = calculate_population_fitness(self.population)
            sorted_population = [x[0] for x in sorted(population_fitness, 
                                                    key=lambda x: x[1], 
                                                    reverse=True)]
            
            # Calculate the number of elites
            num_elites = int(self.elitism_rate * self.population_size)
            elites = sorted_population[:num_elites]
            new_population.extend(elites)
            elite_fitness, elite_components = self.fitness(elites[0])
            print(f"Iteration {iteration}: {len(elites)} elites selected. Elite fitness: {elite_fitness:.4f}")


            # Create a pool for selection, crossover, and mutation that excludes elites
            non_elite_pool = sorted_population[num_elites:]

            crossovers = 0
            mutations = 0

            while len(new_population) < self.population_size:
                if len(non_elite_pool) < 2:
                    # If non-elite pool is too small, replenish it from the entire population
                    non_elite_pool = self.population[:]
                    
                # Selection from non-elite pool
                parent1 = self.select(non_elite_pool)
                parent2 = self.select(non_elite_pool)

                # Crossover and mutation (unchanged)
                child = self.crossover(parent1, parent2)
                if child != parent1 and child != parent2:
                    crossovers += 1

                if random.random() < self.mutation_rate:
                    old_child = child.copy()
                    child = self.mutate(child)
                    if child != old_child:
                        mutations += 1

                new_population.append(child)
                
                # Remove used parents from the non-elite pool to prevent reuse
                if parent1 in non_elite_pool:
                    non_elite_pool.remove(parent1)
                if parent2 in non_elite_pool:
                    non_elite_pool.remove(parent2)

            self.population = new_population
            print(f"Iteration {iteration}: New population created. Crossovers: {crossovers}, Mutations: {mutations}")

            # Clear the fitness cache for the next iteration
            cache_size = len(self.fitness_cache)
            self.fitness_cache.clear()
            print(f"Iteration {iteration}: Fitness cache cleared. Previous cache size: {cache_size}")

            # Evaluate the best fitness in the current population
            best_chromosome = max(self.population, key=lambda x: self.fitness(x)[0])
            best_fitness, best_components = self.fitness(best_chromosome)

            # Add the best fitness of this generation to the history
            fitness_history.append(best_fitness)


            if best_fitness > best_fitness_overall:
                best_fitness_overall = best_fitness
                best_chromosome_overall = best_chromosome

            print(f'Iteration {iteration}: Best Fitness = {best_fitness:.4f}, Overall Best Fitness = {best_fitness_overall:.4f}')
            print("Best chromosome components:")
            for key, value in best_components.items():
                print(f"  {key}: {value:.4f}")

        print("\nGenetic Algorithm Complete!")
        print("Final Best Solution:")
        best_fitness, best_components = self.fitness(best_chromosome_overall)
        print(f"Fitness: {best_fitness:.4f}")
        print("Components:")
        for key, value in best_components.items():
            print(f"  {key}: {value:.4f}")
        
        print("\nBest Chromosome:")
        for service, fog_node_id in best_chromosome_overall.items():
            print(f"  Application {service[0]}, Module {service[1]} -> Fog Node {fog_node_id}")
        self.plot_fitness_progress(fitness_history)


    def output_placement_json(self, chromosome, output_file_path):
        # Convert chromosome to the specified format
        formatted_output = []
        for service, fog_node_id in chromosome.items():
            formatted_output.append({
                "module_name": f"Mod{service[1]}",
                "app": str(service[0]),
                "id_resource": fog_node_id
            })

        # Convert the formatted output to JSON
        placement_json = json.dumps(formatted_output, indent=4)

        # Write to a file
        if os.path.exists(output_file_path):
            return "Error: File already exists."
        else:
            with open(output_file_path, 'w') as file:
                file.write(placement_json)

    #### fixed the Initial placement file json format problem

    def trim_and_output_placement_json(self, chromosome, nodes_df, application_df, output_file_path):
        # Convert chromosome to DataFrame
        chromosome_df = pd.DataFrame(list(chromosome.items()), columns=['Module', 'Node ID'])
        chromosome_df[['Application ID', 'Module ID']] = pd.DataFrame(chromosome_df['Module'].tolist(), index=chromosome_df.index)

        # Merge with the application DataFrame to get RAM requirements
        merged_df = pd.merge(chromosome_df, application_df[['Application ID', 'Module ID', 'Required RAM']], on=['Application ID', 'Module ID'])

        # Make a copy of nodes_df to track available RAM
        nodes_ram_available = nodes_df.copy()
        nodes_ram_available.rename(columns={'RAM': 'Available RAM'}, inplace=True)
        nodes_ram_available.set_index('id', inplace=True)

        # Initialize an empty list for valid placements and counters for failed checks
        valid_placements = []
        failed_modules_count = 0
        failed_apps_set = set()

        # Iterate over each placement
        for index, row in merged_df.iterrows():
            node_id = row['Node ID']
            required_ram = row['Required RAM']
            app_id = row['Application ID']
            module_id = row['Module ID']

            # Check if the node has enough available RAM
            if nodes_ram_available.at[node_id, 'Available RAM'] >= required_ram:
                # Deduct the required RAM from the node's available RAM
                nodes_ram_available.at[node_id, 'Available RAM'] -= required_ram

                # Add to valid placements
                valid_placements.append({
                    "module_name": f"Mod{module_id}",
                    "app": str(app_id),
                    "id_resource": node_id
                })
            else:
                # Increment failed counters
                failed_modules_count += 1
                failed_apps_set.add(app_id)

        # Print the count of failed modules and applications
        print(f"Number of failed module placements: {failed_modules_count}")
        print(f"Number of applications with failed placements: {len(failed_apps_set)}")

        ## adding the Initial placement info
        converted_data = {"initialAllocation": valid_placements}

        # Convert the valid placements to JSON
        placement_json = json.dumps(converted_data, indent=4)
    
        # Write to a file
        if os.path.exists(output_file_path):
            return "Error: File already exists."
        else:
            with open(output_file_path, 'w') as file:
                file.write(placement_json)

        return len(failed_apps_set)


