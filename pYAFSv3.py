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

def KPI_process(folder_file, folderapps, folderpops ,pop_ids, app_ids, method_name, number_of_modules=None):
  # initialize the varibles


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
  lambs = []
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

  modules = []

  # read app event, link, numberofmodules
  data_link = pd.read_csv(folder_file+'_link.csv')
  data_app_event = pd.read_csv(folder_file+'.csv')
  app_modules = read_app_spec(folderapps)
  pop_info = read_pop_spec(folderpops)
  # list all available app_id in the app_event CSV
  app_list = data_app_event["app"].values.tolist()
  app_list = list(dict.fromkeys(app_list))

  for id_app in (app_list):
    # print(f"list app id {id_app}")

    framed_app = data_app_event.loc[data_app_event['app'] == id_app]
    id_app_list = framed_app["id"].values.tolist()
    id_app_list = list(dict.fromkeys(id_app_list))
    # print(f"processing id: {id_app_list}")
    for id_process in id_app_list:
        framed_process = data_app_event.loc[data_app_event['id'] == id_process]
        number_of_module = app_modules.loc[app_modules['name'] == str(id_app), "numberofmodule"].iloc[0]

        if number_of_modules is not None:
            number_of_module = number_of_modules
      # print(f"number of module: {number_of_module}")
        if len(framed_process) >= number_of_module:
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
        ADSA = framed_process.iloc[1:]['time_reception'] - framed_process.iloc[1:]['time_emit']
        ADSA = ADSA.mean()
        total_adsa.append(ADSA)

        #wait collection method 2: using reception - time_in


        # Proses link menggunakan id_process, supaya id yang ga lengkap tidak diproses
        framed_link_id = data_link.loc[data_link['id'] == id_process]
        firstto = framed_link_id[framed_link_id['message'].str.startswith("M.USER.APP.")].shape[0]
        first_mod_hop.append(firstto)
        avg_every_mod = (len(framed_link_id) - firstto) / (number_of_module - 1)
        hop_per_mod.append(avg_every_mod)
        total_hop.append(len(framed_link_id))
        total_bytes.append(sum(framed_link_id['size']))
        total_latency.append(sum(framed_link_id['latency']))
        total_buffer.append(sum(framed_link_id['buffer']))

        # wait dihitung satu per satu
        total_wait.append(total_response - total_service - sum(framed_link_id['latency']))

      # else:
      #   print(f"number of module: {number_of_module}, read length:{len(framed_process)}, link detected: {len(framed_link_id)}")
      #   print("coms not appended")

    if(len(coms) != 0):
        number_coms.append(len(coms))
        average_service_time.append(sum(jumlah_service)/len(coms))
        average_response_time.append(sum(response_total)/len(response_total))
        average_adra.append(sum(total_adra)/len(total_adra))
        average_adsa.append(sum(total_adsa)/len(total_adsa))
        # print(f"average adsa = {dd}, totaladsa: {total_adsa}")
        modules.append(number_of_module)

        if len(total_hop) != 0:
          average_bytes.append(sum(total_bytes)/len(total_bytes))
          average_hop.append(sum(total_hop)/len(total_hop))
          avg_first_mod_hop.append(sum(first_mod_hop)/len(first_mod_hop))
          avg_hop_per_mod.append(sum(hop_per_mod)/len(hop_per_mod))
          average_latency.append(sum(total_latency)/len(total_latency))
          # print(sum(first_mod_hop)/len(first_mod_hop))
          average_buffer.append(sum(total_buffer)/len(total_buffer))
          max_buffer.append(max(total_buffer))
          # wait = bb-aa-(sum(total_latency)/len(total_latency))
          wait = sum(total_wait)/len(total_wait)
          if wait > 0:
            average_wait.append(wait)
          else:
            average_wait.append(0)
        else:
          average_bytes.append(0)
          average_hop.append(0)
          avg_first_mod_hop.append(0)
          avg_hop_per_mod.append(0)
          average_latency.append(0)
          average_buffer.append(0)
          average_wait.append(0)
          max_buffer.append(0)
    else:
      modules.append(number_of_module)
      number_coms.append(0)
      average_service_time.append(0)
      average_response_time.append(0)
      average_adra.append(0)
      average_adsa.append(0)
      average_bytes.append(0)
      average_hop.append(0)
      average_wait.append(0)


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

  trans = pd.DataFrame(list(zip(app_list, number_coms,average_response_time,average_service_time,average_latency,average_adra,average_adsa, average_wait,average_bytes, average_hop, avg_first_mod_hop, avg_hop_per_mod, average_buffer, max_buffer)), columns =['Application ID', 'number of complete comms','avg. total reponse time','avg. total service time', 'avg. total latency', 'ADRA', 'ADSA', 'Wait time','average byte', 'average hop', 'average first hop', 'average hop per mod', 'average total buffer', 'max total buffer'])
  trans['pop_problem_id']=pop_ids
  trans['app_problem_id']=app_ids
  trans['method']=method_name
  df_filtered = trans[trans["number of complete comms"] > 0]
  # trans.to_csv(excel_name, mode='a', index = False)
  # trans.to_csv(excel_name, mode='a', index = False, header =  None)
  print(f"--------------Done for:{pop_ids}_{app_ids}")
  # add_to_csv(trans, excel_name)
  # printcsv(df_filtered)
  # print(df_filtered.describe())

  return df_filtered


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

def create_dataframe_from_json_file(filepath):
    # Load JSON data from the file
    with open(filepath, 'r') as file:
        json_data = json.load(file)

    # Convert links and nodes to DataFrame
    links_df = pd.DataFrame(json_data['links'])
    nodes_df = pd.DataFrame(json_data['nodes'])

    # Rename 'id' column in nodes to 'node_id' for clarity
    nodes_df.rename(columns={'id': 'node_id'}, inplace=True)

    # Left join to add source and target labels
    links_df = links_df.merge(nodes_df[['node_id', 'label']], left_on='source', right_on='node_id', how='left')
    links_df.rename(columns={'label': 'source_label'}, inplace=True)
    links_df.drop('node_id', axis=1, inplace=True)

    links_df = links_df.merge(nodes_df[['node_id', 'label']], left_on='target', right_on='node_id', how='left')
    links_df.rename(columns={'label': 'target_label'}, inplace=True)
    links_df.drop('node_id', axis=1, inplace=True)

    return links_df


# Updating the function to ensure that all four link types are always included in the output DataFrame

def count_link_usage_as_df_updated(csv_filepath, json_filepath, pop_problem_id, app_problem_id, methods):
    link_filepath = f'{csv_filepath}_link.csv'
    
    # Load the CSV file
    csv_df = pd.read_csv(link_filepath)

    # Load the JSON file and categorize links
    with open(json_filepath, 'r') as file:
        json_data = json.load(file)
    json_links = pd.DataFrame(json_data['links'])
    nodes = json_data['nodes']
    for link in json_links.itertuples():
        link_type = further_revised_categorize_link(link._asdict(), nodes)
        json_links.at[link.Index, 'type'] = link_type

    # Create a set of unique link pairs (src, dst) from the CSV file
    csv_link_pairs = set(zip(csv_df['src'], csv_df['dst']))

    # Match these pairs with the links in the JSON file to determine the type of link used
    used_link_types = {}
    for link in json_links.itertuples():
        if (link.source, link.target) in csv_link_pairs or (link.target, link.source) in csv_link_pairs:
            used_link_types[link.type] = used_link_types.get(link.type, 0) + 1

    # Count the total number of each link type in the JSON file
    total_link_type_counts = json_links['type'].value_counts().to_dict()

    # Ensure all four link types are included
    all_link_types = ['node_head_link', 'cloud_link', 'node_link', 'gateway_link']
    flat_data = {}
    for link_type in all_link_types:
        flat_data[f'{link_type}_used'] = used_link_types.get(link_type, 0)
        flat_data[f'{link_type}_total'] = total_link_type_counts.get(link_type, 0)

    # Creating a DataFrame
    link_usage_df = pd.DataFrame([flat_data])
    link_usage_df['pop_problem_id']=pop_problem_id
    link_usage_df['app_problem_id']=app_problem_id
    link_usage_df['method']=methods

    return link_usage_df

def transform_summary_by_label_with_all_labels(df):
    """
    Transforms the DataFrame to summarize total RAM and placement count by label,
    with Application ID as the first column. Ensures inclusion of specific labels.

    Parameters:
    df (DataFrame): The pandas DataFrame to process.

    Returns:
    DataFrame: A transformed DataFrame with Application ID as index and labels as columns.
    """
    # Define the required labels
    all_labels = ['cloud', 'fog_node', 'head_node', 'gateway']

    # Column names in df
    label_column = 'label'
    ram_column = 'Required RAM'
    placement_column = 'Placement Node ID'
    app_id_column = 'Application ID'

    # Calculate total RAM and count of placements
    summary = df.groupby([app_id_column, label_column]).agg({ram_column: 'sum', placement_column: 'count'}).reset_index()

    # Pivot the summary
    ram_pivot = summary.pivot(index=app_id_column, columns=label_column, values=ram_column).reindex(columns=all_labels, fill_value=0)
    count_pivot = summary.pivot(index=app_id_column, columns=label_column, values=placement_column).reindex(columns=all_labels, fill_value=0)

    # Combining the RAM and placement count data
    combined = ram_pivot.join(count_pivot, lsuffix=' Total RAM', rsuffix=' Placement Count')

    return combined

def merge_dataframes(place, app, nodes, pop):
    """
    Merges multiple DataFrames into a single DataFrame based on specified columns.

    Parameters:
    place (DataFrame): DataFrame containing placement data.
    app (DataFrame): DataFrame containing application data.
    nodes (DataFrame): DataFrame containing node data.
    pop (DataFrame): DataFrame containing population data.

    Returns:
    DataFrame: The resulting merged DataFrame.
    """
    result = (
        pd.merge(place, app, on=['Application ID', 'Module Name'], how='left')
        .merge(nodes[['id', 'label']], left_on='Node ID', right_on='id', how='left')
        .drop(columns='id')
        .merge(pop, left_on='Application ID', right_on='app', how='left')
        .rename(columns={'Node ID': 'Placement Node ID', 'id_resource': 'source'})
    )
    return result

def filter_mod0(df):
    """
    Filters the DataFrame for rows where 'Module Name' is 'Mod0'
    and returns the 'Application ID' and 'label' columns.

    Parameters:
    df (DataFrame): The pandas DataFrame to process.

    Returns:
    DataFrame: A DataFrame with filtered rows containing only 'Application ID' and 'label' columns.
    """
    return df[df['Module Name'] == 'Mod0'][['Application ID', 'label', 'source', 'lambda', 'numberofmodule']]

# # Applying the function to the DataFrame
# filtered_mod0_df = filter_mod0(result)
# filtered_mod0_df

def combine_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, columns: list) -> pd.DataFrame:
    # df1['app'] = df1['app'].astype(int)
    # df1['pop_id'] = df1['pop_id'].astype(int)
    # df1['app_id'] = df1['app_id'].astype(int)
    # df2['app'] = df2['app'].astype(int)
    # df2['pop_id'] = df2['pop_id'].astype(int)
    # df2['app_id'] = df2['app_id'].astype(int)
    combined = pd.merge(df1, df2, on=columns, how='left')
    return combined


def resource_analysis(place_folder, pop_folder, app_folder, nodes_folder, app_problem_id, pop_problem_id, method_name):
    place = placement_json_to_dataframe(place_folder)
    app = module_json_to_dataframe(app_folder)
    nodes = nodes_extract_to_dataframe(nodes_folder)
    pop = read_pop_spec(pop_folder)
    result = merge_dataframes(place, app, nodes, pop)

    # Calculate total placement cost per application
    node_costs = nodes[['id', 'cost']].rename(columns={'id': 'Node ID'})
    placement_with_costs = pd.merge(place, node_costs, on='Node ID', how='left')
    total_cost_per_app = placement_with_costs.groupby('Application ID')['cost'].sum().reset_index()
    total_cost_per_app.rename(columns={'cost': 'Total Placement Cost'}, inplace=True)


    transformed_summary_all_labels = transform_summary_by_label_with_all_labels(result)
    filtered_mod0_df = filter_mod0(result)
    output = combine_dataframes(transformed_summary_all_labels, filtered_mod0_df, ['Application ID'])
    
    # Merge total placement cost into the output DataFrame
    output = pd.merge(output, total_cost_per_app, on='Application ID', how='left')

    output['pop_problem_id']=pop_problem_id
    output['app_problem_id']=app_problem_id
    output['method']=method_name
    output.rename(columns = {'label':'first module placement'}, inplace = True)
    return output

def combine_dataframes_save(df1: pd.DataFrame, df2: pd.DataFrame, columns: list) -> pd.DataFrame:
    df1['Application ID'] = df1['Application ID'].astype(int)
    df1['pop_problem_id'] = df1['pop_problem_id'].astype(int)
    df1['app_problem_id'] = df1['app_problem_id'].astype(int)
    df2['Application ID'] = df2['Application ID'].astype(int)
    df2['pop_problem_id'] = df2['pop_problem_id'].astype(int)
    df2['app_problem_id'] = df2['app_problem_id'].astype(int)
    combined = pd.merge(df1, df2, on=columns, how='left')
    return combined

def analyze_placement_result_v2(topo, place, pops, apps, folder_file_result, pop_problem_id, app_problem_id, method, excel_name, save=True):
    # #### RUNNING PROCESS
    KPI = KPI_process(folder_file_result, apps, pops, pop_problem_id, app_problem_id, method)

    resource_result = resource_analysis(place, pops, apps, topo, pop_problem_id, app_problem_id, method)
    # print(resource_result)

    link_usage_counts = count_link_usage_as_df_updated(folder_file_result, topo, pop_problem_id, app_problem_id, method)
    # print(link_usage_counts)


    hasil = combine_dataframes_save(KPI, resource_result, ['Application ID', 'pop_problem_id', 'app_problem_id', 'method'])
    hasil = combine_dataframes(hasil, link_usage_counts, ['pop_problem_id', 'app_problem_id', 'method'] )
    if save == True:
        print("saving file.....")
        save_dataframe_to_csv(hasil, excel_name)
        
    else: 
        print("yeah im not saving that")
        return hasil 

    return 

def analyze_placement_result_v3(topo_folder, place_folder, pops_folder, apps_folder, folder_file_result, pop_problem_id, app_problem_id, size, method, excel_name, save=True):
    # #### RUNNING PROCESS
    KPI = KPI_process(folder_file_result, apps_folder, pops_folder, pop_problem_id, app_problem_id, method)

    resource_result = resource_analysis(place_folder, pops_folder, apps_folder, topo_folder, pop_problem_id, app_problem_id, method)
    # print(resource_result)

    link_usage_counts = count_link_usage_as_df_updated(folder_file_result, topo_folder, pop_problem_id, app_problem_id, method)
    # print(link_usage_counts)


    hasil = combine_dataframes_save(KPI, resource_result, ['Application ID', 'pop_problem_id', 'app_problem_id', 'method'])
    hasil = combine_dataframes(hasil, link_usage_counts, ['pop_problem_id', 'app_problem_id', 'method'] )
    hasil["Problem Size"] = size

    if save == True:
        print("saving file.....")
        save_dataframe_to_csv(hasil, excel_name)
        
    else: 
        print("yeah im not saving that")
        return hasil 

    return 

def KPI_process_and_save (folder_file_result, apps_folder, pops_folder, pop_problem_id, app_problem_id, method, size, excel_name, save=True):
    KPI = KPI_process(folder_file_result, apps_folder, pops_folder, pop_problem_id, app_problem_id, method)
    KPI["Problem Size"] = size
    save_name = f"{excel_name}_KPI.csv"
    if save == True:
        print(f"saving file..... KPI for {pop_problem_id}, {app_problem_id}, {method}, {size}")
        save_dataframe_to_csv(KPI, save_name)
        
    else: 
        print("yeah im not saving that KPI")
        return KPI  
        
def resource_analysis_and_save(place_folder, pops_folder, apps_folder, topo_folder, pop_problem_id, app_problem_id, method, size, excel_name, save=True):
    resource_result = resource_analysis(place_folder, pops_folder, apps_folder, topo_folder, pop_problem_id, app_problem_id, method)
    resource_result["Problem Size"] = size
    save_name = f"{excel_name}_resourceanalysis.csv"
    if save == True:
        print(f"saving file..... resource for {pop_problem_id}, {app_problem_id}, {method}, {size}")
        save_dataframe_to_csv(resource_result, save_name)
        
    else: 
        print("yeah im not saving that Resource") 
        return resource_result

def link_usage_and_save(folder_file_result, topo_folder, pop_problem_id, app_problem_id, method, size, excel_name, save=True): 
    link_usage_counts = count_link_usage_as_df_updated(folder_file_result, topo_folder, pop_problem_id, app_problem_id, method)
    link_usage_counts["Problem Size"] = size
    save_name = f"{excel_name}_linkusage.csv"
    if save == True:
        print(f"saving file..... linkusage for {pop_problem_id}, {app_problem_id}, {method}, {size}")
        save_dataframe_to_csv(link_usage_counts, save_name)
        
    else: 
        print("yeah im not saving that Resource") 
        return link_usage_counts

def analyze_placement_result_v4(topo_folder, place_folder, pops_folder, apps_folder, folder_file_result, pop_problem_id, app_problem_id, size, method, excel_name, save=True):
    KPI = KPI_process_and_save (folder_file_result, apps_folder, pops_folder, pop_problem_id, app_problem_id, method, size, excel_name, save)
    resource_result = resource_analysis_and_save(place_folder, pops_folder, apps_folder, topo_folder, pop_problem_id, app_problem_id, method, size, excel_name, save)    
    link_usage_counts = link_usage_and_save(folder_file_result, topo_folder, pop_problem_id, app_problem_id, method, size, excel_name, save=True)

    
def save_dataframe_to_csv(dataframe, filename):
    import os
    import pandas as pd
    if os.path.exists(filename):
        existing_data = pd.read_csv(filename)
        if list(existing_data.columns) == list(dataframe.columns):
            combined_data = pd.concat([existing_data, dataframe], ignore_index=True)
            combined_data.to_csv(filename, index=False)
            print("Data appended to existing file.")
        else:
            print("Columns of the existing file do not match the DataFrame columns.")
            print(list(existing_data.columns))
            print(list(dataframe.columns))
    else:
        dataframe.to_csv(filename, index=False)
        print("New file created.")

    return True
