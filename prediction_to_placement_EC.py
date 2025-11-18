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
                'Required Storage': module['Storage'],
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

    # Create a deep copy of topology_df to track Storage usage without modifying the original
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
                        # Get the required Storage for this module from the application DataFrame
                        try:
                            required_Storage = app_df[(app_df['Application ID'] == int(app_id)) & 
                                             (app_df['Module ID'] == module_index)]['Required Storage'].values[0]
                        except IndexError:
                            print(f"No matching data found for Application ID: {app_id}, Module Index: {module_index}")
                            print("Available Application IDs:", app_df['Application ID'].unique())
                            print("Available Module IDs:", app_df['Module ID'].unique())
                            raise
                        
                        # Get the available Storage for this resource from the topology DataFrame
                        resource_id_int = int(resource_id[1:])  # Remove 'n' prefix and convert to int
                        available_Storage = working_topology_df[working_topology_df['id'] == resource_id_int]['Storage'].values[0]
                        
                        # Check if there's enough Storage available for this module
                        if required_Storage <= available_Storage:
                            # Create allocation entry
                            allocation = {
                                "module_name": f"Mod{module_index}",
                                "app": app_id,
                                "id_resource": resource_id_int
                            }
                            app_allocations.append(allocation)
                        else:
                            # Not enough Storage, mark placement as failed
                            app_placement_success = False
                            failure_reason = f"Not enough Storage for module {module_index} on resource {resource_id_int}"
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
            # Only subtract Storage if the application is successfully placed
            for allocation in app_allocations:
                resource_id = allocation['id_resource']
                module_name = allocation['module_name']
                required_Storage = app_df[(app_df['Application ID'] == int(app_id)) & 
                                      (app_df['Module Name'] == module_name)]['Required Storage'].values[0]
                working_topology_df.loc[working_topology_df['id'] == resource_id, 'Storage'] -= required_Storage
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
