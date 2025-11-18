import json
import random
from random import randrange
import pandas as pd
import numpy as np
import os

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


# Function to convert placement JSON to a DataFrame
def placement_json_to_dataframe(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    rows = []
    for placement in data["initialAllocation"]:
        rows.append({
            'Module Name': placement['module_name'],
            'Application ID': str(placement['app']),
            'Node ID': placement['id_resource']
        })
    return pd.DataFrame(rows)


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
                'Message ID': message['id'],
                'Message Name': message['name'],
                'Bytes': message['bytes'],
                'Instructions': message['instructions']
            }
            rows.append(row)

    # Convert the list of rows into a DataFrame
    df = pd.DataFrame(rows)

    return df

def read_app_spec(folderapps):

  import pandas as pd
  import json
  with open(folderapps) as f:
      data = json.load(f)

  # Convert the dictionary into a DataFrame
  df = pd.DataFrame(data)

  # Extract 'name' and 'numberofmodule' columns from the DataFrame
  df_extracted = df[['name', 'numberofmodule']]

  # Display the extracted DataFrame
  # print(df_extracted.head())
  return df_extracted

def read_pop_spec(folderpops):
  import pandas as pd
  import json
  with open(folderpops) as f:
      data = json.load(f)
  df = pd.DataFrame(data)
  # Explore the 'sources' key
  sources_data = df['sources']

  # Convert this to a DataFrame
  df_sources = pd.json_normalize(sources_data)
  return df_sources

import pandas as pd # Ensure pandas is imported

def KPI_processv3(folder_file, folderapps, folderpops, place_folder_name, nodes_folder_name, pop_ids, app_ids, method_name, number_of_modules=None):
  # initialize the varibles

  # Energy calculation parameters
  cloud_bw = 125000  # Cloud bandwidth
  edge_bw = 75000    # Edge bandwidth
  edge_tx_factor = 5.5  # Edge transmission power factor
  edge_rx_factor = 4.5  # Edge reception power factor
  cloud_tx_factor = 40  # Cloud transmission power factor
  cloud_rx_factor = 30  # Cloud reception power factor
  cloud_processing_factor = 250  # Cloud processing power factor
  edge_processing_factor = 65    # Edge processing power factor
  cloud_node_id = 100  # Cloud node ID

  list_of_complete =[]
  number_coms = []
  coms = []
  jumlah_service = []
  average_service_time = []
  response_total = []
  average_response_time = []
  average_adra= []
  total_adra = []
  average_adsa = []
  total_adsa = []
  first_mod_hop = []
  hop_per_mod = []
  avg_first_mod_hop = []
  avg_hop_per_mod = []

  total_waitmethod2 = []
  lambda_values = [] # List to store lambda values for each app
  sourced = []

  total_bytes = []
  average_bytes = []
  total_hop = []
  average_hop = []

  total_latency = []
  average_latency = []

  total_buffer = []
  average_buffer = []
  max_buffer = []
  m_buffer = []

  total_wait = []
  average_wait =[]

  average_waitmethod2 = []

  modules = [] # List to store the number of modules for each app

  # Energy metrics
  total_comm_energy = []
  total_proc_energy = []
  total_energy = []

  # read app event, link, numberofmodules
  data_link = pd.read_csv(folder_file+'_link.csv')
  data_app_event = pd.read_csv(folder_file+'.csv')
  app_modules = read_app_spec(folderapps)
  pop_info = read_pop_spec(folderpops) # pop_info has 'app' and 'lambda' columns

  # --- Calculate Placement Cost ---
  # Assuming placement_json_to_dataframe and nodes_extract_to_dataframe are defined elsewhere
  try:
      place = placement_json_to_dataframe(place_folder_name)
      nodes = nodes_extract_to_dataframe(nodes_folder_name)
      node_costs = nodes[['id', 'Cost']].rename(columns={'id': 'Node ID'})
      placement_with_costs = pd.merge(place, node_costs, on='Node ID', how='left')
      # Ensure 'Application ID' is the correct type for merging later (assuming int)
      placement_with_costs['Application ID'] = placement_with_costs['Application ID'].astype(int)
      total_cost_per_app = placement_with_costs.groupby('Application ID')['Cost'].sum().reset_index()
      # Rename for merging
      total_cost_per_app = total_cost_per_app.rename(columns={'Application ID': 'app_id', 'Cost': 'total_placement_cost'})
      cost_calculated = True
  except Exception as e:
      print(f"Warning: Could not calculate placement cost. Error: {e}")
      total_cost_per_app = pd.DataFrame(columns=['app_id', 'total_placement_cost']) # Create empty df
      cost_calculated = False


  # list all available app_id in the app_event CSV
  # Ensure app_id is integer type for consistency
  data_app_event['app'] = data_app_event['app'].astype(int)
  app_list = data_app_event["app"].unique().tolist()
  app_list = sorted(app_list) # Sort to ensure consistent order

  for id_app in (app_list):
    # print(f"list app id {id_app}")

    framed_app = data_app_event.loc[data_app_event['app'] == id_app]
    # Ensure 'id' (process id) is treated consistently if needed, assuming it's okay as is
    id_app_list = framed_app["id"].unique().tolist()
    # print(f"processing id: {id_app_list}")

    # Determine number_of_module for the current app *before* the inner loop
    current_number_of_module = None
    try:
        # Assuming app_modules 'name' column stores app_id as string
        current_number_of_module = app_modules.loc[app_modules['name'] == str(id_app), "numberofmodule"].iloc[0]
        if number_of_modules is not None: # Override if global value is provided
            current_number_of_module = number_of_modules
    except IndexError:
        print(f"Warning: Number of modules not found for app {id_app} in app_modules.")
        if number_of_modules is not None:
             current_number_of_module = number_of_modules
        # Handle case where module count isn't found and no override is given (e.g., set to default or skip app)


    for id_process in id_app_list:
        framed_process = data_app_event.loc[data_app_event['id'] == id_process]
        # Use the pre-determined current_number_of_module
        if current_number_of_module is None:
            print(f"Skipping process {id_process} for app {id_app} due to missing module count.")
            continue # Skip this process if module count is unknown

      # print(f"number of module: {number_of_module}")
        if len(framed_process) >= current_number_of_module:
            framed_process = framed_process.sort_values(by=['time_emit'])
            # only complete coms are recorded
            coms.append(id_process)

            total_service = framed_process['service'].sum()
            jumlah_service.append(total_service)

            max_time_out = framed_process.iloc[-1]['time_out']
            min_time_in = framed_process.iloc[0]['time_emit']
            total_response = max_time_out - min_time_in
            response_total.append(total_response)

            # Calculate ADRA (Average Delay from Requester to Application)
            first_row = framed_process.iloc[0]
            ADRA = first_row['time_reception'] - first_row['time_emit']
            # wait2 = first_row['time_in'] - first_row['time_reception']
            # total_waitmethod2.append(wait2)
            total_adra.append(ADRA)


            # Calculate ADSA (Average Delay of the Services)
            # Ensure there are subsequent rows before calculating ADSA
            if len(framed_process) > 1:
                ADSA = framed_process.iloc[1:]['time_reception'] - framed_process.iloc[1:]['time_emit']
                ADSA = ADSA.mean() # Calculate mean directly on the Series
                total_adsa.append(ADSA)
            else:
                total_adsa.append(0) # Or np.nan or handle as appropriate

            #wait collection method 2: using reception - time_in


            # Proses link menggunakan id_process, supaya id yang ga lengkap tidak diproses
            framed_link_id = data_link.loc[data_link['id'] == id_process]
            firstto = framed_link_id[framed_link_id['message'].str.startswith("M.USER.APP.")].shape[0]
            first_mod_hop.append(firstto)
            # Avoid division by zero if number_of_module is 1
            if current_number_of_module > 1:
                # Ensure denominator is not zero before division
                denominator = current_number_of_module - 1
                avg_every_mod = (len(framed_link_id) - firstto) / denominator if denominator != 0 else 0
            else:
                avg_every_mod = 0 # Or handle as appropriate if only one module
            hop_per_mod.append(avg_every_mod)
            total_hop.append(len(framed_link_id))
            total_bytes.append(sum(framed_link_id['size']))
            total_latency.append(sum(framed_link_id['latency']))
            total_buffer.append(sum(framed_link_id['buffer']))

            # wait dihitung satu per satu
            # Ensure framed_link_id is not empty before summing latency
            link_latency_sum = sum(framed_link_id['latency']) if not framed_link_id.empty else 0
            total_wait.append(total_response - total_service - link_latency_sum)

            # Calculate energy metrics for this process
            # Communication energy calculation
            process_bytes = sum(framed_link_id['size'])
            is_cloud_link = (framed_link_id['src'] == cloud_node_id) | (framed_link_id['dst'] == cloud_node_id)
            transmission_time = np.where(
                is_cloud_link,
                process_bytes / cloud_bw,
                process_bytes / edge_bw
            )
            
            # Calculate transmission and reception power
            is_source_cloud = framed_link_id['src'] == cloud_node_id
            is_dest_cloud = framed_link_id['dst'] == cloud_node_id
            
            tx_power = np.where(
                is_source_cloud,
                transmission_time * cloud_tx_factor,
                transmission_time * edge_tx_factor
            )
            
            rx_power = np.where(
                is_dest_cloud,
                transmission_time * cloud_rx_factor,
                transmission_time * edge_rx_factor
            )
            
            process_comm_energy = sum(tx_power + rx_power)
            total_comm_energy.append(process_comm_energy)
            
            # Processing energy calculation
            cloud_service_time = framed_process.loc[framed_process['TOPO.dst'] == cloud_node_id, 'service'].sum()
            edge_service_time = framed_process.loc[framed_process['TOPO.dst'] != cloud_node_id, 'service'].sum()
            
            process_proc_energy = (cloud_service_time * cloud_processing_factor) + (edge_service_time * edge_processing_factor)
            total_proc_energy.append(process_proc_energy)
            
            # Total energy for this process
            process_total_energy = process_comm_energy + process_proc_energy
            total_energy.append(process_total_energy)

        # else:
        #   print(f"number of module: {current_number_of_module}, read length:{len(framed_process)}, link detected: {len(framed_link_id)}")
        #   print("coms not appended")

    # --- Calculations and appends for the current id_app ---
    current_lambda = None # Default value if not found or no complete comms
    modules.append(current_number_of_module) # Append the module count for this app

    if(len(coms) != 0):
        num_valid_coms = len(coms)
        number_coms.append(num_valid_coms)
        average_service_time.append(sum(jumlah_service)/num_valid_coms)
        average_response_time.append(sum(response_total)/num_valid_coms) # Use num_valid_coms
        average_adra.append(sum(total_adra)/num_valid_coms) # Use num_valid_coms
        # Ensure total_adsa is not empty before calculating mean
        avg_adsa_val = sum(total_adsa)/num_valid_coms if total_adsa else 0 # Use num_valid_coms
        average_adsa.append(avg_adsa_val)
        # print(f"average adsa = {dd}, totaladsa: {total_adsa}")
        # modules.append(current_number_of_module) # Append module count here

        # Check len(total_hop) which should be equal to num_valid_coms if calculated correctly inside loop
        if len(total_hop) == num_valid_coms:
          average_bytes.append(sum(total_bytes)/num_valid_coms)
          average_hop.append(sum(total_hop)/num_valid_coms)
          avg_first_mod_hop.append(sum(first_mod_hop)/num_valid_coms)
          avg_hop_per_mod.append(sum(hop_per_mod)/num_valid_coms)
          average_latency.append(sum(total_latency)/num_valid_coms)
          # print(sum(first_mod_hop)/len(first_mod_hop))
          average_buffer.append(sum(total_buffer)/num_valid_coms)
          max_buffer.append(max(total_buffer)) # Max over the processes for this app
          # wait = bb-aa-(sum(total_latency)/len(total_latency))
          wait = sum(total_wait)/num_valid_coms
          average_wait.append(max(0, wait)) # Ensure wait time is not negative
          
          # Add average energy metrics
          average_comm_energy = sum(total_comm_energy)/num_valid_coms
          average_proc_energy = sum(total_proc_energy)/num_valid_coms
          average_total_energy = sum(total_energy)/num_valid_coms
        else: # Should ideally not happen if logic is correct, but handle defensively
          print(f"Warning: Mismatch between len(coms)={num_valid_coms} and len(total_hop)={len(total_hop)} for app {id_app}")
          average_bytes.append(0)
          average_hop.append(0)
          avg_first_mod_hop.append(0)
          avg_hop_per_mod.append(0)
          average_latency.append(0)
          average_buffer.append(0)
          average_wait.append(0)
          max_buffer.append(0)
          average_comm_energy = 0
          average_proc_energy = 0
          average_total_energy = 0

        # Get lambda for this app_id
        try:
            # Assuming 'app' column in pop_info corresponds to id_app (as string)
            # And 'lambda' column holds the value. Adjust column names if different.
            current_lambda = pop_info.loc[pop_info['app'] == str(id_app), 'lambda'].iloc[0]
        except (IndexError, KeyError):
            print(f"Warning: Lambda value not found for app {id_app} in pop_info.")
            current_lambda = None # Or 0 or np.nan

    else: # No complete communications for this app
      # Still need to append placeholders to keep lists aligned
      # modules.append(current_number_of_module) # Already appended above
      number_coms.append(0)
      average_service_time.append(0)
      average_response_time.append(0)
      average_adra.append(0)
      average_adsa.append(0)
      average_bytes.append(0)
      average_hop.append(0)
      avg_first_mod_hop.append(0)
      avg_hop_per_mod.append(0)
      average_latency.append(0)
      average_buffer.append(0)
      average_wait.append(0)
      max_buffer.append(0)
      # Also append lambda (or placeholder) even if no comms
      try:
          # Assuming 'app' column in pop_info stores app_id as string
          current_lambda = pop_info.loc[pop_info['app'] == str(id_app), 'lambda'].iloc[0]
      except (IndexError, KeyError):
          print(f"Warning: Lambda value not found for app {id_app} in pop_info.")
          current_lambda = None # Or 0 or np.nan

    lambda_values.append(current_lambda) # Append lambda for the current app_id

    # --- Reset lists for the next id_app ---
    coms = []
    jumlah_service = []
    response_total = []
    total_adra = []
    total_adsa = []
    total_hop=[]
    first_mod_hop = []
    hop_per_mod = []

    total_bytes=[]
    total_latency=[]
    total_buffer = []
    total_wait = []

  # --- Create final DataFrame ---
  # Ensure all lists have the same length before creating the DataFrame
  list_lengths = {
      "app_list": len(app_list), "modules": len(modules), "number_coms": len(number_coms),
      "avg_resp_time": len(average_response_time), "avg_serv_time": len(average_service_time),
      "avg_latency": len(average_latency), "avg_adra": len(average_adra), "avg_adsa": len(average_adsa),
      "avg_wait": len(average_wait), "avg_bytes": len(average_bytes), "avg_hop": len(average_hop),
      "avg_first_hop": len(avg_first_mod_hop), "avg_hop_per_mod": len(avg_hop_per_mod),
      "avg_buffer": len(average_buffer), "max_buffer": len(max_buffer), "lambda": len(lambda_values)
  }
  # Check if all lengths are equal to len(app_list)
  if not all(length == len(app_list) for length in list_lengths.values()):
      print("Error: Mismatch in list lengths before creating DataFrame. Check logic.")
      # Print lengths for debugging
      print(f"Lengths: {list_lengths}")
      # You might want to raise an error or return an empty DataFrame here
      return pd.DataFrame()


  trans = pd.DataFrame(list(zip(app_list, modules, number_coms, average_response_time,
                              average_service_time, average_latency, average_adra,
                              average_adsa, average_wait, average_bytes, average_hop,
                              avg_first_mod_hop, avg_hop_per_mod, average_buffer,
                              max_buffer, lambda_values, average_comm_energy,
                              average_proc_energy, average_total_energy)),
                      columns=['app_id', 'number_of_modules', 'throughput',
                              'app_response_time', 'app_service_time', 'app_latency',
                              'delay_from_request', 'delay_between_module', 'wait_time',
                              'total_byte', 'total_hop', 'first_hop_distance',
                              'hop_per_mod', 'application_total_buffer',
                              'application_max_total_buffer', 'request_lambda',
                              'communication_energy', 'processing_energy', 'total_energy'])

  # Ensure app_id in trans is integer for merging
  trans['app_id'] = trans['app_id'].astype(int)

  # Merge with cost data if it was calculated successfully
  if cost_calculated:
      trans = pd.merge(trans, total_cost_per_app, on='app_id', how='left')
      # Fill NaN costs with 0 or another placeholder if needed
      trans['total_placement_cost'] = trans['total_placement_cost'].fillna(0)
  else:
      # Add the cost column with a default value (e.g., 0 or NaN) if cost calculation failed
      trans['total_placement_cost'] = 0 # Or np.nan

  trans['pop_problem_id']=pop_ids
  trans['app_problem_id']=app_ids
  trans['placement_algorithm']=method_name

  # Filter out rows where throughput is 0 (no complete communications)
  df_filtered = trans[trans["throughput"] > 0].copy() # Use .copy() to avoid SettingWithCopyWarning

  # trans.to_csv(excel_name, mode='a', index = False)
  # trans.to_csv(excel_name, mode='a', index = False, header =  None)
  print(f"--------------Done for:{pop_ids}_{app_ids}")
  # add_to_csv(trans, excel_name)
  # printcsv(df_filtered)
  # print(df_filtered.describe())

  return df_filtered
import os # Ensure os is imported for file existence check

def process_and_save_kpi(folder_file, folderapps, folderpops, place_folder_name,
                         nodes_folder_name, pop_ids, app_ids, method_name,
                         output_csv_path, number_of_modules=None, mode='append'):
    """
    Processes KPI data using KPI_processv3 based on simulation results, saves the result to a CSV file,
    and returns the processed DataFrame.

    Args:
        folder_file (str): Path to the simulation result folder (containing results.csv, etc.).
        folderapps (str): Path to the application specification JSON file.
        folderpops (str): Path to the population specification JSON file (containing lambda values).
        place_folder_name (str): Path to the placement JSON file.
        nodes_folder_name (str): Path to the topology/nodes JSON file.
        pop_ids (int): Population problem ID.
        app_ids (int): Application problem ID (or dataset ID).
        method_name (str): Name of the placement algorithm used.
        output_csv_path (str or os.PathLike): Path where the output CSV file will be saved.
        number_of_modules (optional): Specific number of modules to consider (passed to KPI_processv3). Defaults to None.
        mode (str): The mode for saving the CSV file.
                    'append': Adds the new data as rows. Creates the file with headers if it doesn't exist. (Default)
                    'rewrite': Overwrites the file with the new data and headers.

    Returns:
        pd.DataFrame or None: The processed DataFrame containing KPIs if processing is successful
                              (this DataFrame can be empty if no valid KPIs were found).
                              Returns None if an error occurred during processing or if
                              KPI_processv3 itself returned None.

    Raises:
        ValueError: If an invalid mode is provided.
        TypeError: If output_csv_path is not a string or PathLike object.
    """
    print(f"Starting KPI processing for pop={pop_ids}, app={app_ids}, method={method_name}...")
    print(f"  Simulation results: {folder_file}")
    print(f"  Application spec: {folderapps}")
    print(f"  Population spec: {folderpops}")
    print(f"  Placement file: {place_folder_name}")
    print(f"  Nodes file: {nodes_folder_name}")
    print(f"  Output CSV: {output_csv_path} (Mode: {mode})")

    # --- Check type of output_csv_path before proceeding ---
    # This addresses the error "expected str, bytes or os.PathLike object, not tuple"
    if not isinstance(output_csv_path, (str, os.PathLike)):
        print(f"Error: Invalid type for output_csv_path. Expected string or PathLike, but got {type(output_csv_path)}: {output_csv_path}")
        # Optionally raise a TypeError instead of just printing and returning None
        # raise TypeError(f"output_csv_path must be a string or PathLike object, but got {type(output_csv_path)}")
        return None # Return None as the path is invalid for file operations
    # --- End of type check ---

    df_processed = None # Initialize df_processed to None

    try:
        # Call the main processing function
        df_processed = KPI_processv3(
            folder_file=folder_file,
            folderapps=folderapps,
            folderpops=folderpops,
            place_folder_name=place_folder_name,
            nodes_folder_name=nodes_folder_name,
            pop_ids=pop_ids,
            app_ids=app_ids,
            method_name=method_name,
            number_of_modules=number_of_modules
        )

        # Check if the result is valid before saving
        if df_processed is not None:
            if not df_processed.empty:
                print(f"KPI processing successful. Saving results to {output_csv_path} in '{mode}' mode...")

                # Save the DataFrame to CSV based on the specified mode
                if mode == 'append':
                    # Check if file exists to determine if header should be written
                    # The type check above ensures output_csv_path is valid here
                    header = not os.path.exists(output_csv_path)
                    df_processed.to_csv(output_csv_path, mode='a', header=header, index=False)
                    print(f"Successfully appended KPI data to {output_csv_path}")
                elif mode == 'rewrite':
                    df_processed.to_csv(output_csv_path, mode='w', header=True, index=False)
                    print(f"Successfully rewrote KPI data to {output_csv_path}")
                else:
                    # Raise an error for invalid modes
                    raise ValueError(f"Invalid mode '{mode}'. Choose 'append' or 'rewrite'.")

            else:
                # DataFrame is empty, but processing was successful
                print(f"Warning: KPI processing for pop={pop_ids}, app={app_ids} resulted in an empty DataFrame. No CSV file saved.")
            # Return the DataFrame (either populated or empty)
            return df_processed
        else:
            # This case occurs if KPI_processv3 explicitly returned None
            print(f"Warning: KPI_processv3 returned None for pop={pop_ids}, app={app_ids}. No CSV file saved.")
            return None # Return None as KPI_processv3 indicated failure/no result

    except FileNotFoundError as fnf_error:
        print(f"Error: Input file not found during KPI processing for pop={pop_ids}, app={app_ids}. Details: {fnf_error}")
        return None # Return None on error
    except KeyError as key_error:
         print(f"Error: Missing expected key in input data during KPI processing for pop={pop_ids}, app={app_ids}. Details: {key_error}")
         return None # Return None on error
    except ValueError as ve: # Catch the specific ValueError for invalid mode or others from KPI_processv3/pandas
        print(f"ValueError during KPI processing or saving for pop={pop_ids}, app={app_ids}: {ve}")
        return None # Return None on value error
    except TypeError as te: # Catch unexpected TypeErrors that might still occur
        print(f"TypeError during KPI processing or saving for pop={pop_ids}, app={app_ids}: {te}")
        # Log traceback for debugging if this happens despite the initial check
        # import traceback
        # print(traceback.format_exc())
        return None # Return None on type error
    except Exception as e:
        # Catch any other unexpected errors during processing or saving
        print(f"An unexpected error occurred during KPI processing or saving for pop={pop_ids}, app={app_ids}: {e} (Type: {type(e).__name__})")
        # Consider logging the full traceback here for debugging if needed
        # import traceback
        # print(traceback.format_exc())
        return None # Return None on error

import pandas as pd
import numpy as np

def calculate_communication_energy(file_path, 
                                    cloud_node_id=100, 
                                    cloud_bw=125000, 
                                    edge_bw=75000, 
                                    edge_tx_factor=5.5, 
                                    edge_rx_factor=4.5, 
                                    cloud_tx_factor=40, 
                                    cloud_rx_factor=30):
    """
    Reads link data, calculates transmission time, transmission power, 
    receiving power, and total communication power.

    Args:
        file_path (str): Path to the link data CSV file.
        cloud_node_id (int): The ID of the cloud node.
        cloud_bw (int): Bandwidth for links involving the cloud node.
        edge_bw (int): Bandwidth for links between edge/fog nodes.
        edge_tx_factor (float): Power factor for transmission from non-cloud nodes.
        edge_rx_factor (float): Power factor for reception at non-cloud nodes.
        cloud_tx_factor (float): Power factor for transmission from the cloud node.
        cloud_rx_factor (float): Power factor for reception at the cloud node.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The DataFrame with added metrics columns.
            - float: The total communication power.
    """
    # Read the data
    data_link = pd.read_csv(file_path + "_link.csv")

    # Calculate transmission time using vectorized approach
    is_cloud_link = (data_link['src'] == cloud_node_id) | (data_link['dst'] == cloud_node_id)
    data_link['transmission_time'] = np.where(
        is_cloud_link,
        data_link['size'] / cloud_bw, # Value if True (Cloud link)
        data_link['size'] / edge_bw   # Value if False (Edge link)
    )

    # Display the dataframe with the new column to verify
    print("Transmission Time Calculation:")
    print(data_link[['src', 'dst', 'size', 'transmission_time']].head())
    print("-" * 30)

    # Calculate Transmission Power based *only* on the source node
    is_source_cloud = data_link['src'] == cloud_node_id
    data_link['transmission_power'] = np.where(
        is_source_cloud,
        data_link['transmission_time'] * cloud_tx_factor, # Value if True (Cloud Tx)
        data_link['transmission_time'] * edge_tx_factor   # Value if False (Edge Tx)
    )

    # Calculate Receiving Power based *only* on the destination node
    is_dest_cloud = data_link['dst'] == cloud_node_id
    data_link['receiving_power'] = np.where(
        is_dest_cloud,
        data_link['transmission_time'] * cloud_rx_factor, # Value if True (Cloud Rx)
        data_link['transmission_time'] * edge_rx_factor    # Value if False (Edge Rx)
    )

    # Calculate total communication power per link
    data_link['communication_power'] = data_link['transmission_power'] + data_link['receiving_power']

    # Display the relevant columns to verify
    print("Power Calculation:")
    print(data_link[['src', 'dst', 'size', 'transmission_time', 'transmission_power', 'receiving_power', 'communication_power']].head())
    print("-" * 30)

    # Calculate the total communication power
    total_communication_power = data_link['communication_power'].sum()

    # Print the total communication power
    print(f"Total Communication Power: {total_communication_power}")
    
    return total_communication_power

# Example usage: Assuming 'folder_file' variable is defined in a previous cell
# Make sure folder_file is defined before calling this function.
# Example: folder_file = 'path/to/your/data/prefix' 
# data_link_updated, total_comm_power = calculate_communication_metrics(folder_file + '_link.csv')

# # If you want to run it directly here (replace with actual path if needed):
# # Assuming folder_file is defined in the notebook's global scope
# try:
#     data_link_updated, total_comm_power = calculate_communication_metrics(folder_file + '_link.csv')
# except NameError:
#     print("Error: 'folder_file' variable is not defined.")
#     print("Please define 'folder_file' with the path prefix for your CSV files.")
#     # Example: folder_file = '../data/simulation_results/scenario1' 
#     data_link_updated, total_comm_power = (None, None) # Assign None if folder_file is missing

import numpy as np # Assuming numpy is needed and might not be imported yet in this specific cell
import pandas as pd # Assuming pandas might be needed for DataFrame operations
import os # Import os for file path operations
from typing import Tuple # Import Tuple for type hinting

def calculate_processing_and_idle_power(foldername: str,
                                        cloud_node_id: int = 100,
                                        cloud_processing_factor: int = 250,
                                        edge_processing_factor: int = 65,
                                        total_simulation_time: int = 20000,
                                        number_of_non_cloud_devices: int = 100,
                                        cloud_idle_power_factor: int = 100,
                                        non_cloud_idle_power_factor: int = 25) -> Tuple[float, float]:
    """
    Reads application event data, calculates processing power for each event,
    calculates total idle power based on service times and simulation duration,
    and returns both total processing power and total idle power.

    Args:
        foldername: The base path/prefix for the CSV file (e.g., 'path/to/results').
                    The function will read '{foldername}.csv'.
        cloud_node_id: The ID representing the cloud node.
        cloud_processing_factor: Factor to multiply service time for cloud processing power.
        edge_processing_factor: Factor to multiply service time for edge/fog processing power.
        total_simulation_time: The total duration of the simulation.
        number_of_non_cloud_devices: The count of devices excluding the cloud.
        cloud_idle_power_factor: Factor to multiply cloud idle time for idle power.
        non_cloud_idle_power_factor: Factor to multiply non-cloud idle time for idle power.

    Returns:
        A tuple containing:
            - total_processing_power (float): Sum of processing power across all events.
            - total_idle_power (float): Sum of idle power for cloud and non-cloud nodes.

    Raises:
        FileNotFoundError: If the CSV file '{foldername}.csv' does not exist.
        ValueError: If the loaded DataFrame does not contain 'TOPO.dst' or 'service' columns.
        Exception: For other potential pandas read_csv errors.
    """
    file_path = f"{foldername}.csv"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")

    try:
        data_app_event = pd.read_csv(file_path)
        print(f"Successfully read data from {file_path}")
    except Exception as e:
        print(f"Error reading CSV file '{file_path}': {e}")
        raise # Re-raise the exception after printing

    if 'TOPO.dst' not in data_app_event.columns or 'service' not in data_app_event.columns:
        raise ValueError(f"DataFrame loaded from '{file_path}' must contain 'TOPO.dst' and 'service' columns.")

    # --- Calculate Processing Power ---
    data_app_event['processing_power'] = np.where(
        data_app_event['TOPO.dst'] == cloud_node_id,                # Condition: Is destination the cloud?
        data_app_event['service'] * cloud_processing_factor,      # Value if True (Cloud Processing)
        data_app_event['service'] * edge_processing_factor        # Value if False (Edge/Fog Processing)
    )
    total_processing_power = data_app_event['processing_power'].sum()

    # Display processing power calculation details
    print("\nProcessing Power Calculation (Head):")
    print(data_app_event[['TOPO.dst', 'service', 'processing_power']].head())
    print(f"Total Processing Power: {total_processing_power}")
    print("-" * 30)

    # --- Calculate Idle Power ---
    # Calculate total service time for the cloud node
    total_cloud_service_time = data_app_event.loc[data_app_event['TOPO.dst'] == cloud_node_id, 'service'].sum()

    # Calculate cloud idle time and power
    cloud_idle_time = max(0, total_simulation_time - total_cloud_service_time)
    cloud_idle_power = cloud_idle_time * cloud_idle_power_factor

    # Calculate total service time for non-cloud nodes
    total_non_cloud_service_time = data_app_event.loc[data_app_event['TOPO.dst'] != cloud_node_id, 'service'].sum()

    # Calculate total potential time for non-cloud nodes
    total_non_cloud_potential_time = total_simulation_time * number_of_non_cloud_devices

    # Calculate non-cloud idle time and power
    non_cloud_idle_time = max(0, total_non_cloud_potential_time - total_non_cloud_service_time)
    non_cloud_idle_power = non_cloud_idle_time * non_cloud_idle_power_factor

    # Calculate total idle power
    total_idle_power = cloud_idle_power + non_cloud_idle_power

    # Print idle power calculation details
    print("Idle Power Calculation:")
    print(f"Total Cloud Service Time: {total_cloud_service_time}")
    print(f"Cloud Idle Time: {cloud_idle_time}")
    print(f"Cloud Idle Power: {cloud_idle_power}")
    print("-" * 15)
    print(f"Total Non-Cloud Service Time: {total_non_cloud_service_time}")
    print(f"Total Non-Cloud Potential Time: {total_non_cloud_potential_time}")
    print(f"Non-Cloud Idle Time: {non_cloud_idle_time}")
    print(f"Non-Cloud Idle Power: {non_cloud_idle_power}")
    print("-" * 15)
    print(f"Total Idle Power: {total_idle_power}")
    print("-" * 30)

    # Return both total processing power and total idle power
    return total_processing_power, total_idle_power

# Example usage (assuming folder_file variable holds the path prefix):
# try:
#     # Define factors and parameters if different from defaults
#     # cloud_proc_factor = 250
#     # edge_proc_factor = 65
#     # cloud_id = 100
#     # sim_time = 20000
#     # non_cloud_count = 100
#     # cloud_idle_factor = 100
#     # non_cloud_idle_factor = 25
#
#     # Make sure 'folder_file' is defined, e.g.:
#     # folder_file = '../data/simulation_results/scenario1'
#
#     total_proc_power, total_idle_pwr = calculate_processing_and_idle_power(
#         folder_file # Pass the folder/file prefix here
#         # cloud_node_id=cloud_id,
#         # cloud_processing_factor=cloud_proc_factor,
#         # edge_processing_factor=edge_proc_factor,
#         # total_simulation_time=sim_time,
#         # number_of_non_cloud_devices=non_cloud_count,
#         # cloud_idle_power_factor=cloud_idle_factor,
#         # non_cloud_idle_power_factor=non_cloud_idle_factor
#     )
#     print(f"Function returned:")
#     print(f"  Total Processing Power: {total_proc_power}")
#     print(f"  Total Idle Power: {total_idle_pwr}")
#
# except NameError:
#      print("Error: 'folder_file' variable is not defined.")
#      total_proc_power, total_idle_pwr = None, None
# except FileNotFoundError as e:
#      print(e) # Print the specific file not found error
#      total_proc_power, total_idle_pwr = None, None
# except ValueError as e:
#      print(e) # Print the specific column missing error
#      total_proc_power, total_idle_pwr = None, None
# except Exception as e:
#      print(f"An unexpected error occurred: {e}")
#      total_proc_power, total_idle_pwr = None, None

import pandas as pd
import os

# Assuming calculate_communication_energy and calculate_processing_and_idle_power
# are defined elsewhere and accessible in this scope.

def total_energy_scenario(folder_file, pop_problem_id, app_problem_id, placement_algorithm, output_csv_path='energy_summary.csv', mode='append'):
    """
    Calculates communication, processing, and idling energy based on simulation files,
    saves the results along with problem identifiers and placement algorithm to a CSV file,
    and returns the calculated energy values.

    Args:
        folder_file (str): Path prefix for simulation result files (e.g., '../data/sim_results/run1').
                           Used by the helper calculation functions.
        pop_problem_id: Identifier for the population problem configuration.
        app_problem_id: Identifier for the specific application instance within the population.
        placement_algorithm (str): Name of the placement algorithm used for this scenario.
        output_csv_path (str): The path to the CSV file where results will be saved.
                               Defaults to 'energy_summary.csv'.
        mode (str): The mode for saving the CSV file.
                    'append': Adds the new data as a row. Creates the file with headers if it doesn't exist. (Default)
                    'rewrite': Overwrites the file with the new data and headers.

    Returns:
        tuple: A tuple containing (communication_energy, processing_energy, idling_energy, total_energy).
               Returns (None, None, None, None) if any calculation step fails or an error occurs.

    Raises:
        ValueError: If an invalid mode is provided.
    """
    try:
        # Calculate energy components using helper functions
        communication = calculate_communication_energy(folder_file)
        pros, idling = calculate_processing_and_idle_power(folder_file)

        # Check if calculations were successful (assuming they return None on failure)
        if communication is None or pros is None or idling is None:
            print(f"Warning: Energy calculation failed for {folder_file}. Skipping CSV save.")
            return None, None, None, None

        total_energy = communication + pros + idling

        # Prepare data for the DataFrame
        data = {
            'pop_problem_id': [pop_problem_id],
            'app_problem_id': [app_problem_id],
            'placement_algorithm': [placement_algorithm],
            'communication_energy': [communication],
            'processing_energy': [pros],
            'idling_energy': [idling],
            'total_energy': [total_energy]
        }
        df = pd.DataFrame(data)

        # Save the DataFrame to CSV based on the specified mode
        if mode == 'append':
            # Check if file exists to determine if header should be written
            header = not os.path.exists(output_csv_path)
            df.to_csv(output_csv_path, mode='a', header=header, index=False)
            # print(f"Appended energy results to {output_csv_path}") # Optional: for debugging
        elif mode == 'rewrite':
            df.to_csv(output_csv_path, mode='w', header=True, index=False)
            # print(f"Rewrote energy results to {output_csv_path}") # Optional: for debugging
        else:
            # Raise an error for invalid modes
            raise ValueError(f"Invalid mode '{mode}'. Choose 'append' or 'rewrite'.")

        # Return the calculated values
        return communication, pros, idling, total_energy

    except FileNotFoundError as e:
        print(f"Error: A required file was not found during energy calculation for {folder_file}. Details: {e}")
        return None, None, None, None
    except Exception as e:
        # Catch other potential errors during calculation or file writing
        print(f"An unexpected error occurred for {folder_file}: {e}")
        return None, None, None, None