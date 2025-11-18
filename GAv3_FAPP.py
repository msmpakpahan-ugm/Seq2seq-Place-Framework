import pandas as pd
import json
import networkx as nx 
import random
import numpy as np
import os


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



class GeneticAlgorithmv3:
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
        # # Parse the JSON files into DataFrames


        # # # Diagnostic prints to check DataFrame structures
        # print("Placement DataFrame Columns:", placement_df.columns)
        # print("Application DataFrame Columns:", application_df.columns)
        # print("Data types in Placement DataFrame:", placement_df.dtypes)
        # print("Data types in Application DataFrame:", application_df.dtypes)

        # Merge and calculate RAM usage
        placement_app_merged = pd.merge(placement_df, application_df, on=['Module Name', 'Application ID'], how='left')
        node_ram_usage = placement_app_merged.groupby('Node ID')['Required RAM'].sum().reset_index()
        nodes_df.rename(columns={'RAM': 'Available RAM'}, inplace=True)

        # Compare available and required RAM
        node_ram_comparison = pd.merge(nodes_df, node_ram_usage, left_on='id', right_on='Node ID', how='left')
        node_ram_comparison['Required RAM'].fillna(0, inplace=True)
        node_ram_comparison['RAM_Sufficient'] = node_ram_comparison['Available RAM'] >= node_ram_comparison['Required RAM']

        # print(node_ram_comparison[['id', 'Available RAM', 'Required RAM', 'RAM_Sufficient']])
        # Check if all nodes have sufficient RAM and print a message if so
        if node_ram_comparison['RAM_Sufficient'].all():
            # print("All is good.")
            return True
        else:
            # print("Something not fit")
            return False

    def print_placement_validity (self, nodes_df, application_df, placement_df):
        # Merge and calculate RAM usage
        placement_app_merged = pd.merge(placement_df, application_df, on=['Module Name', 'Application ID'], how='left')
        node_ram_usage = placement_app_merged.groupby('Node ID')['Required RAM'].sum().reset_index()
        nodes_df.rename(columns={'RAM': 'Available RAM'}, inplace=True)

        # Compare available and required RAM
        node_ram_comparison = pd.merge(nodes_df, node_ram_usage, left_on='id', right_on='Node ID', how='left')
        node_ram_comparison['Required RAM'].fillna(0, inplace=True)
        node_ram_comparison['RAM_Sufficient'] = node_ram_comparison['Available RAM'] >= node_ram_comparison['Required RAM']

        return node_ram_comparison[['id', 'Available RAM', 'Required RAM', 'RAM_Sufficient']]


    def fitness(self, chromosome):
            # Using a tuple representation of the chromosome as a hashable key for caching
            chromosome_key = tuple(chromosome.items())
            if chromosome_key in self.fitness_cache:
                return self.fitness_cache[chromosome_key]

            total_hops = 0
            num_applications = len(self.app_to_source_node)

            # Optimized: Using set comprehension directly
            for app_id in {app['Application ID'] for _, app in self.app_df.iterrows()}:
                app_total_hops = 0

                # Optimized: Using list comprehension and reducing DataFrame access
                modules = sorted([module for module in chromosome if module[0] == app_id], key=lambda x: x[1])

                if modules:
                    source_node_id = self.app_to_source_node[str(app_id)]
                    app_total_hops += self.shortest_paths_table.at[source_node_id, chromosome[modules[0]]]

                    for i in range(1, len(modules)):
                        app_total_hops += self.shortest_paths_table.at[chromosome[modules[i - 1]], chromosome[modules[i]]]

                total_hops += app_total_hops

            placement_df = self.chromosome_to_dataframe(chromosome, self.app_df)
            all_fit = self.check_placement_validity(self.nodes_df, self.app_df, placement_df)
            # average_hops = total_hops / num_applications if num_applications > 0 else float('inf')
            # fitness_score = 1 / average_hops if average_hops > 0 else float('inf')
            # fitness_score += 0.3 if all_fit else -0.3
            fitness_score = 1 / total_hops if total_hops > 0 else float('inf')
            fitness_score += 0.3 if all_fit else -0.3

            # Caching the calculated fitness value
            self.fitness_cache[chromosome_key] = fitness_score
            return fitness_score


    def select(self):
        # Tournament selection
        tournament_size = max(5, len(self.population))
        tournament = random.sample(self.population, tournament_size)
        tournament.sort(key=lambda chromo: self.fitness(chromo))
        return tournament[0]

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

        for iteration in range(num_iterations):
            new_population = []
            # Optimized: Avoid sorting the entire population when only elites are needed
            sorted_population = sorted(self.population, key=self.fitness, reverse=True)[:int(self.elitism_rate * self.population_size)]
            elites = sorted_population
            new_population.extend(elites)
            print(f"Iteration {iteration}: {len(elites)} elites selected. Elite fitness: {self.fitness(elites[0]):.4f}")

            crossovers = 0
            mutations = 0

            while len(new_population) < self.population_size:
                # Selection - select two parents
                parent1 = self.select()
                parent2 = self.select()

                # Crossover - create a child by crossing over parents
                child = self.crossover(parent1, parent2)
                if child != parent1 and child != parent2:
                    crossovers += 1

                # Mutation - potentially mutate the child
                if random.random() < self.mutation_rate:
                    old_child = child.copy()
                    child = self.mutate(child)
                    if child != old_child:
                        mutations += 1

                # Add the new child to the new population
                new_population.append(child)

            # Replace the old population with the new one
            self.population = new_population
            print(f"Iteration {iteration}: New population created. Crossovers: {crossovers}, Mutations: {mutations}")

            # Clear the fitness cache for the next iteration
            cache_size = len(self.fitness_cache)
            self.fitness_cache.clear()
            print(f"Iteration {iteration}: Fitness cache cleared. Previous cache size: {cache_size}")

            # Evaluate the best fitness in the current population
            best_chromosome = max(self.population, key=self.fitness)
            best_fitness = self.fitness(best_chromosome)
            if best_fitness > best_fitness_overall:
                best_fitness_overall = best_fitness
                best_chromosome_overall = best_chromosome

            print(f'Iteration {iteration}: Best Fitness = {best_fitness:.4f}, Overall Best Fitness = {best_fitness_overall:.4f}')
            print(f"Best chromosome: {dict(list(best_chromosome.items())[:5])}...")  # Show first 5 items
            print(f"Iteration {iteration} completed.")

        print("Genetic Algorithm run completed.")
        print(f"Final Best Fitness: {best_fitness_overall:.4f}")
        print(f"Final Best Chromosome: {dict(list(best_chromosome_overall.items())[:10])}...")  # Show first 10 items

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

