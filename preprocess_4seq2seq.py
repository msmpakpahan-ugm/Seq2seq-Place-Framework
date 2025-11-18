import random
import pandas as pd
import json

def node_specification(file_path):
    """
    Read node specifications from a JSON file and return a DataFrame.

    Args:
    file_path (str): Path to the JSON file containing node data.

    Returns:
    pd.DataFrame: DataFrame with columns 'id', 'IPT', 'RAM', and 'Bandwidth'.
    """
    with open(file_path, 'r') as file:
        data_topology = json.load(file)
    
    node_data = []
    for node in data_topology['nodes']:
        node_data.append({
            'node_id': node['id'],
            'IPT': node['IPT'],
            'RAM': node['RAM'],
            'type': node['label']  # Use the Bandwidth value from the node data
        })

    return pd.DataFrame(node_data)

def app_specification(file_path):
    """
    Read application specifications from a JSON file and return a DataFrame.

    Args:
    file_path (str): Path to the JSON file containing application data.

    Returns:
    pd.DataFrame: DataFrame with application specifications.
    """
    with open(file_path, 'r') as file:
        data_app = json.load(file)

    module_data = []
    message_data = []

    for app in data_app:
        app_name = app['name']
        deadline = app['deadline']
        number_of_modules = app['numberofmodule']

        for module in app['module']:
            module_data.append({
                'app_id': app_name,
                'module_id': module['name'].replace("'", ""),
                'RAM': module['RAM'],
                'deadline': deadline,
                'number_of_module': number_of_modules
            })

        for message in app['message']:
            message_data.append({
                'Message': message['name'],
                'bytes': message['bytes'],
                'instruction': message['instructions']
            })

    df_modules = pd.DataFrame(module_data)
    df_message = pd.DataFrame(message_data)
    
    final_app = pd.concat([df_modules, df_message], axis=1)
    return final_app

def create_placement_dataframe(placement_file_path):
    with open(placement_file_path, 'r') as file:
        placement_data = json.load(file)

    # Extracting placement information
    apps = []
    modules = []
    nodes = []

    for allocation in placement_data['initialAllocation']:
        apps.append(allocation['app'])
        modules.append(allocation['module_name'])
        nodes.append(allocation['id_resource'])

    # Creating DataFrame
    placement_df = pd.DataFrame({
        'app_id': apps,
        'module_id': modules,
        'placement_node': nodes
    })

    return placement_df

def create_population_dataframe(population_file_path):
    with open(population_file_path, 'r') as f:
        data_pop = json.load(f)
    
    id_resource = []
    app = []
    message = []
    lambda_val = []
    
    for source in data_pop['sources']:
        id_resource.append(source['id_resource'])
        app.append(source['app'])
        message.append(source['message'])
        lambda_val.append(source['lambda'])
    
    df_population = pd.DataFrame({
        'source_id': id_resource,
        'app_id': app,
        'message': message,
        'lambda': lambda_val
    })
    
    return df_population

def merge_dataframes(pop_df, app_df, place_df):
    # Merge pop_df with app_df and place_df based on app_id and module_id
    merged_df = pop_df.merge(app_df, left_on='app_id', right_on='app_id', how='left')
    merged_df = merged_df.merge(place_df, left_on=['app_id', 'module_id'], right_on=['app_id', 'module_id'], how='left')
    
    # Display the first few rows of the merged dataframe
    print(merged_df.head())
    
    # Display information about the merged dataframe
    print(merged_df.info())
    
    # You might want to save this merged dataframe for further use
    # merged_df.to_csv('merged_data.csv', index=False)
    
    return merged_df

def create_txt_file(merged_df, file):
    # Group the dataframe by app_id
    grouped = merged_df.groupby('app_id')
    
    for _, app_group in grouped:
        # First line: app info
        first_row = app_group.iloc[0]
        app_line = ""
        for _, row in app_group.iterrows():
            module_ram = row['RAM']
            module_message = row['bytes']
            app_line += f"{module_ram} {module_message} "
        app_line += f"n{first_row['source_id']}"
        file.write(app_line.strip() + '\n')
        
        # Second line: placement info
        placement_line = ""
        for _, row in app_group.iterrows():
            placement_line += f"n{row['placement_node']} "
        file.write(placement_line.strip() + '\n')
        
        # Add a blank line between applications
        # if app_id != grouped.ngroups - 1:  # If it's not the last group
            # file.write('\n')

def previous_format_txt(app_path, place_path, pop_path, output_path, combine=False):
    app_df = app_specification(app_path)
    place_df = create_placement_dataframe(place_path) 
    pop_df = create_population_dataframe(pop_path) 
    merged_df = merge_dataframes(pop_df, app_df, place_df)

    mode = 'a' if combine else 'w'

    with open(output_path, mode) as f:
        # if combine:
        #     f.write('\n')
        create_txt_file(merged_df, f)

    print(f"{'Updated' if combine else 'Created'} sequence file: {output_path}")

def create_txt_noIPT(merged_df, file):
    # Group the dataframe by app_id
    grouped = merged_df.groupby('app_id')
    
    for app_id, app_group in grouped:
        # First line: Order Number (app_id), Number of Module, RAM values, Source
        first_row = app_group.iloc[0]
        num_modules = len(app_group)
        ram_values = ' '.join(f"r{row['RAM']}" for _, row in app_group.iterrows())
        source = first_row['source_id']
        
        first_line = f"o{app_id} m{num_modules} {ram_values} n{source}\n"
        file.write(first_line)
        
        # Second line: Id Node Target values
        placement_line = ' '.join(f"n{row['placement_node']}" for _, row in app_group.iterrows())
        file.write(placement_line.strip() + '\n')

def noIPT_format_txt(app_path, place_path, pop_path, output_path, combine=False):
    app_df = app_specification(app_path)
    place_df = create_placement_dataframe(place_path) 
    pop_df = create_population_dataframe(pop_path) 
    merged_df = merge_dataframes(pop_df, app_df, place_df)

    mode = 'a' if combine else 'w'

    with open(output_path, mode) as f:
        create_txt_noIPT(merged_df, f)

    print(f"{'Updated' if combine else 'Created'} sequence file: {output_path}")

def create_txt_IPT(merged_df, file):
    # Group the dataframe by app_id
    grouped = merged_df.groupby('app_id')
    
    for app_id, app_group in grouped:
        # First line: Order Number (app_id), Number of Module, RAM values, Source
        first_row = app_group.iloc[0]
        num_modules = len(app_group)
        ram_values = ' '.join(f"r{row['RAM']}" for _, row in app_group.iterrows())
        source = first_row['source_id']
        ipt_values = ' '.join(f"i{row['instruction']}" for _, row in app_group.iterrows())
        
        first_line = f"o{app_id} m{num_modules} {ram_values} {ipt_values} n{source}\n"
        file.write(first_line)
        
        # Second line: Id Node Target values
        placement_line = ' '.join(f"n{row['placement_node']}" for _, row in app_group.iterrows())
        file.write(placement_line.strip() + '\n')

def IPT_format_txt(app_path, place_path, pop_path, output_path, combine=False):
    app_df = app_specification(app_path)
    place_df = create_placement_dataframe(place_path) 
    pop_df = create_population_dataframe(pop_path) 
    merged_df = merge_dataframes(pop_df, app_df, place_df)

    mode = 'a' if combine else 'w'

    with open(output_path, mode) as f:
        create_txt_IPT(merged_df, f)

    print(f"{'Updated' if combine else 'Created'} sequence file: {output_path}")


def create_test_csv_withIPT(app_path, pop_path, output_path):
    # Load application data
    app_df = app_specification(app_path)
    
    # Load population data
    pop_df = create_population_dataframe(pop_path)

    # Merge app_df and pop_df
    merged_df = pop_df.merge(app_df, left_on='app_id', right_on='app_id', how='left')
    print(merged_df)
    
    grouped = merged_df.groupby('app_id')
    
    data = []
    for app_id, app_group in grouped:
        num_modules = len(app_group)
        ram_values = ' '.join(f"r{row['RAM']}" for _, row in app_group.iterrows())
        source = app_group.iloc[0]['source_id']
        lambda_value = app_group.iloc[0]['lambda']
        ipt_values = ' '.join(f"i{row['instruction']}" for _, row in app_group.iterrows())
        
        modified_input = f"o{app_id} m{num_modules} {ram_values} {ipt_values} l{lambda_value} n{source}"
        
        data.append({
            'app_id': app_id,
            'number_of_modules': num_modules,
            'modified_input': modified_input
        })
    
    # Create DataFrame, sort by app_id as integer, and write to CSV
    df = pd.DataFrame(data)
    df['app_id'] = df['app_id'].astype(int)  # Convert app_id to integer
    df_sorted = df.sort_values(by='app_id')
    df_sorted.to_csv(output_path, index=False)
    print(f"CSV file created: {output_path}")


def create_test_csv_noIPT(app_path, pop_path, output_path):
    # Load application data
    app_df = app_specification(app_path)
    
    # Load population data
    pop_df = create_population_dataframe(pop_path)

    # Merge app_df and pop_df
    merged_df = pop_df.merge(app_df, left_on='app_id', right_on='app_id', how='left')
    # print(merged_df)
    
    grouped = merged_df.groupby('app_id')
    
    data = []
    for app_id, app_group in grouped:
        num_modules = len(app_group)
        ram_values = ' '.join(f"r{row['RAM']}" for _, row in app_group.iterrows())
        source = app_group.iloc[0]['source_id']
        # ipt_values = ' '.join(f"i{row['instruction']}" for _, row in app_group.iterrows())
        
        modified_input = f"o{app_id} m{num_modules} {ram_values} n{source}"
        
        data.append({
            'app_id': app_id,
            'number_of_modules': num_modules,
            'modified_input': modified_input
        })
    
    # Create DataFrame, sort by app_id as integer, and write to CSV
    df = pd.DataFrame(data)
    df['app_id'] = df['app_id'].astype(int)  # Convert app_id to integer
    df_sorted = df.sort_values(by='app_id')
    df_sorted.to_csv(output_path, index=False)
    print(f"CSV file created: {output_path}")


def create_test_csv_previousmodule(app_path, pop_path, output_path):
    # Load application data
    app_df = app_specification(app_path)
    
    # Load population data
    pop_df = create_population_dataframe(pop_path)

    # Merge app_df and pop_df
    merged_df = pop_df.merge(app_df, left_on='app_id', right_on='app_id', how='left')
    # print(merged_df)
    
    grouped = merged_df.groupby('app_id')
    
    data = []
    for app_id, app_group in grouped:
        # First line: app info
        num_modules = len(app_group)

        app_line = ""
        for _, row in app_group.iterrows():
            module_ram = row['RAM']
            module_message = row['bytes']
            app_line += f"{module_ram} {module_message} "
        app_line += f"n{app_group.iloc[0]['source_id']}"
        
        data.append({
            'app_id': app_id,
            'number_of_modules': num_modules,
            'modified_input': app_line
        })
    
    # Create DataFrame, sort by app_id as integer, and write to CSV
    df = pd.DataFrame(data)
    df['app_id'] = df['app_id'].astype(int)  # Convert app_id to integer
    df_sorted = df.sort_values(by='app_id')
    df_sorted.to_csv(output_path, index=False)
    print(f"CSV file created: {output_path}")


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

import pandas as pd
import json
import copy

def create_placement_json(input_csv, output_json, application_json, topology_json):
    # Read the CSV file containing predictions
    df = pd.read_csv(input_csv)

    # Convert application JSON to DataFrame for easier processing
    app_df = json_to_dataframe(application_json)

    # Convert topology JSON to DataFrame for easier processing
    topology_df = extract_nodes_to_dataframe(topology_json)

    # Initialize the list to store all successful allocations
    initial_allocation = []

    # Initialize counters for tracking placement success
    successful_placements = 0
    total_applications = len(df)

    # Create a deep copy of topology_df to track RAM usage without modifying the original
    working_topology_df = copy.deepcopy(topology_df)

    # Iterate through each row in the dataframe (each row represents an application)
    for _, row in df.iterrows():
        app_id = str(row['app_id'])
        num_modules = row['number_of_modules']
        output_words = row['final_output'].split()
        
        app_placement_success = True
        app_allocations = []
        failure_reason = ""

        # Create module allocations for this app
        for module_index in range(num_modules):
            if module_index < len(output_words):
                resource_id = output_words[module_index]
                if resource_id != '<EOS>' and resource_id != '' and resource_id != '<UNK>':
                    try:
                        # Get the required RAM for this module from the application DataFrame
                        required_ram = app_df[(app_df['Application ID'] == int(app_id)) & 
                                              (app_df['Module ID'] == module_index)]['Required RAM'].values[0]
                        
                        # Get the available RAM for this resource from the topology DataFrame
                        resource_id_int = int(resource_id[1:])  # Remove 'n' prefix and convert to int
                        available_ram = working_topology_df[working_topology_df['id'] == resource_id_int]['RAM'].values[0]
                        
                        # Check if there's enough RAM available for this module
                        if required_ram <= available_ram:
                            # Create allocation entry
                            allocation = {
                                "module_name": f"Mod{module_index}",
                                "app": app_id,
                                "id_resource": resource_id_int
                            }
                            app_allocations.append(allocation)
                        else:
                            # Not enough RAM, mark placement as failed
                            app_placement_success = False
                            failure_reason = f"Not enough RAM for module {module_index} on resource {resource_id_int}"
                            break
                    except ValueError:
                        # Invalid resource ID, mark placement as failed
                        app_placement_success = False
                        failure_reason = f"Invalid resource ID for module {module_index}"
                        break
                else:
                    # Invalid resource ID, mark placement as failed
                    app_placement_success = False
                    failure_reason = f"Invalid resource ID for module {module_index}"
                    break
            else:
                # Not enough predicted placements, mark as failed
                app_placement_success = False
                failure_reason = "Not enough predicted placements for all modules"
                break

        if app_placement_success:
            # If all modules were placed successfully, update counters and add to initial_allocation
            successful_placements += 1
            initial_allocation.extend(app_allocations)
            # Only subtract RAM if the application is successfully placed
            for allocation in app_allocations:
                resource_id = allocation['id_resource']
                module_name = allocation['module_name']
                required_ram = app_df[(app_df['Application ID'] == int(app_id)) & 
                                      (app_df['Module Name'] == module_name)]['Required RAM'].values[0]
                working_topology_df.loc[working_topology_df['id'] == resource_id, 'RAM'] -= required_ram
            print(f"Application {app_id} successfully placed.")
        else:
            print(f"Application {app_id} placement failed. Reason: {failure_reason}")

    # Calculate percentage of successful placements
    success_percentage = (successful_placements / total_applications) * 100 if total_applications > 0 else 0

    # Create the final JSON with all successful placements
    placement_data = {
        "initialAllocation": initial_allocation
    }

    import os

    # Check if the file already exists
    if not os.path.exists(output_json):
        # Write to JSON file
        with open(output_json, 'w') as f:
            json.dump(placement_data, f, indent=2)

        print(f"Placement JSON file has been created successfully: {output_json}")
    else:
        print(f"File already exists, skipping writing JSON: {output_json}")

    print(f"Percentage of successful placements: {success_percentage:.2f}%")

    return success_percentage

# Example usage:
# success_rate = create_placement_json('input.csv', 'placement_prediction.json', 'application.json', 'topology.json')
