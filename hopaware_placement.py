import random
import pandas as pd
import json


def node_specification(data_topologi):
    ipt =[]
    ram =[]
    #storage=[]
    id_node=[]
    bandwith_node = []

    for n in data_topologi['nodes']:
        ipt.append(n['IPT'])
        ram.append(n['RAM'])
        #storage.append(n['Storage'])
        id_node.append(n['id'])
        bandwith_node.append(75000)

    df_nodes = pd.DataFrame(list(zip(id_node, ipt, ram, bandwith_node)), columns =['id', 'IPT', 'RAM', 'Bandwith'])
    return df_nodes

def app_specification(data_app):
    name_app_module=[]
    ram_module=[]
    name_module=[]

    name_app_message=[]
    bytes_size=[]
    instruction=[]
    name_message=[]

    deadline=[]

    for k in data_app:
        name_app = k['name']
        for l in k['module']:
            s = l['name']
            namanya = s.replace("'", "")
            ram_module.append(l['RAM'])
            name_module.append(namanya)
            name_app_module.append(name_app)
            deadline.append(k['deadline'])

        for m in k['message']:
            name_app_message.append(name_app)
            name_message.append(m['name'])
            bytes_size.append(m['bytes'])
            instruction.append(m['instructions'])

    df_modules = pd.DataFrame(list(zip(name_app_module, name_module, ram_module, deadline)), columns =['app', 'Module', 'RAM', 'deadline'])
    df_message = pd.DataFrame(list(zip(name_message, bytes_size, instruction)), columns =['Message', 'bytes', 'instruction'])
    final_app = pd.concat([df_modules, df_message], axis=1)
    return final_app

def create_dict(topo):
    new_dict = dict()
    source = []
    for i in topo['links']:
        source.append(i['source'])
        if i['source'] in new_dict:
            # append the new number to the existing array at this slot
            new_dict[i['source']].append(i['target'])
            if i['target'] in new_dict:
                new_dict[i['target']].append(i['source'])
            else:
                new_dict[i['target']] = [i['source']]
        else:
            # create a new array in this slot
            new_dict[i['source']] = [i['target']]
    return new_dict, source

def checkthisout(place, appp, topo):
    # nanti masukin pathnya aja
    ## APP input


    f = open(str(place))
    data_placement = json.load(f)

    df_nodes = topo


    #framing placement
    app =[]
    Mod =[]
    node = []

    for n in data_placement['initialAllocation']:
        app.append(n['app'])
        Mod.append(n['module_name'])
        node.append(n['id_resource'])
    data_popo = pd.DataFrame(list(zip(app, Mod, node)), columns =['App', 'Mod', 'node'])

    df_apps = appp
    muatan = []
    for index, row in data_popo.iterrows():
        current_mod = df_apps.loc[(df_apps['app'] == str(row['App']))&(df_apps['Module'] == str(row['Mod']))]
        df_nodes.loc[row['node'], 'RAM'] = int(df_nodes.loc[row['node'], 'RAM']) - int(current_mod['RAM'])
        # df_nodes.loc[row['node'], 'IPT'] = int(df_nodes.loc[row['node'], 'IPT']) - int(current_mod['Instruction'])

        # df_nodes.loc[row['node'], 'Bandwith'] = int(df_nodes.loc[row['node'], 'Bandwith']) - int(current_mod['Bandwith'])

        # if((df_nodes.loc[row['node'], 'IPT'] <= 0) or (df_nodes.loc[row['node'], 'RAM'] <= 0) or (df_nodes.loc[row['node'], 'Bandwith'] <= 0)):
        #     muatan.append('Tidak Muat')
        # else:
        #     muatan.append('Muat')

        if(df_nodes.loc[row['node'], 'RAM'] <= 0):
            muatan.append('Tidak Muat')
        else:
            muatan.append('Muat')
    return muatan

def createlistofplacement(linkplacement):
    f = open(str(linkplacement))
    data_placement = json.load(f)

    #framing placement
    app =[]
    Mod =[]
    node = []

    for n in data_placement['initialAllocation']:
        app.append(n['app'])
        Mod.append(n['module_name'])
        node.append(n['id_resource'])
    data_popo = pd.DataFrame(list(zip(app, Mod, node)), columns =['App', 'Mod', 'node'])
    return data_popo

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

def placement_json_to_dataframe(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    rows = []
    for placement in data['initialAllocation']:
        rows.append({
            'Module Name': str(placement['module_name']),
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

    # Merge and calculate RAM usage
    placement_app_merged = pd.merge(placement_df, application_df, on=['Module Name', 'Application ID'], how='left')
    node_ram_usage = placement_app_merged.groupby('Node ID')['Required RAM'].sum().reset_index()
    nodes_df.rename(columns={'RAM': 'Available RAM'}, inplace=True)

    # Compare available and required RAM
    node_ram_comparison = pd.merge(nodes_df, node_ram_usage, left_on='id', right_on='Node ID', how='left')
    node_ram_comparison['Required RAM'].fillna(0, inplace=True)
    node_ram_comparison['RAM_Sufficient'] = node_ram_comparison['Available RAM'] >= node_ram_comparison['Required RAM']

       # Check if all nodes have sufficient RAM and print a message if so
    if node_ram_comparison['RAM_Sufficient'].all():
        print("All is good.")
        return node_ram_comparison[['id', 'Available RAM', 'Required RAM', 'RAM_Sufficient']]
    else:
        print("Something not fit")
        return node_ram_comparison[['id', 'Available RAM', 'Required RAM', 'RAM_Sufficient']]

def create_placementold(data_pop, new_dict, app_spec, node_spec, popul, app, save_folder, hopnumber):
    JSONfile = {"initialAllocation": []}
    allocation = []

    allocation_df = pd.DataFrame(columns=['source_id', 'module_name', 'app_id', 'id_resource'])

    for i in data_pop['sources']:

        if hopnumber == 3:
            pilihan = random.choice(new_dict[i['id_resource']])
            print("HOP3 algorithm called")
        elif hopnumber == 2:
            pilihan = i['id_resource']
            print("HOP2 algorithm called")

        applik = i['app']
        current_app = app_spec.loc[app_spec['app'] == applik]

        # print("Source : ", i['id_resource'])
        print("Aplikasi : ", i['app'])
        for index, row in current_app.iterrows():
            ketemu = False
            count_first = 0

            # if row['Module'] == "Mod0":
            #     pilihan = i['id_resource']

            while count_first < 3:
                string1 = {}
                # current_pilihan = pilihan
                if not ketemu:
                    count = 0

                    while count < 5:
                        ram_pilihan = int(node_spec.loc[int(pilihan), 'RAM']) - int(row['RAM'])
                        penentuan_mod = row['Module']


                        if pilihan == 100:
                            string1 = {
                                "module_name": str(row['Module']),
                                "app": str(applik),
                                "id_resource": pilihan
                            }
                            ketemu = True
                            print(f"found resource on cloud: {pilihan}")
                            break
                        elif ram_pilihan >= 0:
                            string1 = {
                                "module_name": str(row['Module']),
                                "app": str(applik),
                                "id_resource": pilihan
                            }
                            print(f"found resource on node {pilihan}")
                            node_spec.loc[int(pilihan), 'RAM'] = ram_pilihan
                            pilihan = random.choice(new_dict[pilihan])
                            print(f"next selection based on {pilihan}")
                            ketemu = True
                            break
                        else:
                            if row['Module'] == "Mod0":
                                pilihan = random.choice(new_dict[i['id_resource']])
                            else:
                                pilihan = random.choice(new_dict[pilihan])
                                print(f"current pilihan{pilihan} selection: {new_dict[pilihan]}")
                        count += 1
                    if ketemu:
                        print(f"we out from break inside {row['Module']}")
                        break
                else:
                    pilihan = random.choice(new_dict[pilihan])
                    print(f"failed to find node next to for {count}, finding new in{pilihan}")
                    break
                count_first += 1

            print("Count first : ", count_first)
            if count_first == 3 or pilihan == 100:
                print(f"this loop if fail to placing {count_first} or cloud because no placement {pilihan}")
                pilihan = 100
                string1 = {
                    "module_name": str(row['Module']),
                    "app": str(applik),
                    "id_resource": 100
                }


            allocation.append(string1)

            # # Store data in the DataFrame
            # allocation_df = allocation_df.append({
            #     'source_id': i['id_resource'],
            #     'module_name': row['Module'],
            #     'app_id': applik,
            #     'id_resource': string1['id_resource']
            # }, ignore_index=True)

    JSONfile["initialAllocation"] += allocation
    json_object = json.dumps(JSONfile, indent=2)

    with open(f"{save_folder}/{popul}_{app}_hop{hopnumber}.json", "w") as outfile:
        outfile.write(json_object)

    return True

# You can call the function like this:
# allocation_data = create_placement(data_pop, new_dict, app_spec, node_spec, popul, generate)
# allocation_data.to_csv("allocation_data.csv", index=False)  # Save the DataFrame to a CSV file

def hopaware_txt_noIPT(data_pop, new_dict, app_spec, node_spec, popul, app, save_folder, hopnumber):
    output_lines = []
    order = 0  # Initialize order counter
    
    for i in data_pop['sources']:
        if hopnumber == 3:
            pilihan = random.choice(new_dict[i['id_resource']])
        elif hopnumber == 2:
            pilihan = i['id_resource']

        applik = i['app']
        current_app = app_spec.loc[app_spec['app'] == applik]

        # Count the number of modules
        module_count = len(current_app)

        placement_line = []
        spec_line = [f"o{order}", f"m{module_count}"]  # Start with order and module count
        failed_nodes = []  # List to store failed node IDs

        for _, row in current_app.iterrows():
            spec_line.append(f"r{row['RAM']}")  # Add RAM value
            
            ketemu = False
            count_first = 0

            while count_first < 3 and not ketemu:
                if pilihan == 100 or count_first == 2:
                    placement_line.append(f"n100")
                    ketemu = True
                else:
                    ram_pilihan = int(node_spec.loc[int(pilihan), 'RAM']) - int(row['RAM'])
                    if ram_pilihan >= 0:
                        placement_line.append(f"n{pilihan}")
                        node_spec.loc[int(pilihan), 'RAM'] = ram_pilihan
                        pilihan = random.choice(new_dict[pilihan])
                        ketemu = True
                    else:
                        failed_nodes.append(f"f{pilihan}")  # Add failed node to the list
                        pilihan = random.choice(new_dict[pilihan])
                count_first += 1

        # Add source node to spec_line
        spec_line.append(f"n{i['id_resource']}")
        
        # Add failed nodes to spec_line
        spec_line.extend(failed_nodes)

        # Add the specification line and placement line to the output
        output_lines.append(" ".join(spec_line))
        output_lines.append(" ".join(placement_line))

        order += 1  # Increment order for next application

    # Write the output to a file in append mode
    with open(f"{save_folder}/hop{hopnumber}_newformat_withfailure.txt", "a") as outfile:
        outfile.write("\n".join(output_lines) + "\n")  # Add newline at the end to separate from previous content

    return True


def create_placementold(data_pop, new_dict, app_spec, node_spec, popul, app, save_folder, hopnumber):
    JSONfile = {"initialAllocation": []}
    allocation = []

    allocation_df = pd.DataFrame(columns=['source_id', 'module_name', 'app_id', 'id_resource'])

    for i in data_pop['sources']:

        if hopnumber == 3:
            pilihan = random.choice(new_dict[i['id_resource']])
            print("HOP3 algorithm called")
        elif hopnumber == 2:
            pilihan = i['id_resource']
            print("HOP2 algorithm called")

        applik = i['app']
        current_app = app_spec.loc[app_spec['app'] == applik]

        # print("Source : ", i['id_resource'])
        print("Aplikasi : ", i['app'])
        for index, row in current_app.iterrows():
            ketemu = False
            count_first = 0

            # if row['Module'] == "Mod0":
            #     pilihan = i['id_resource']

            while count_first < 3:
                string1 = {}
                # current_pilihan = pilihan
                if not ketemu:
                    count = 0

                    while count < 5:
                        ram_pilihan = int(node_spec.loc[int(pilihan), 'RAM']) - int(row['RAM'])
                        penentuan_mod = row['Module']


                        if pilihan == 100:
                            string1 = {
                                "module_name": str(row['Module']),
                                "app": str(applik),
                                "id_resource": pilihan
                            }
                            ketemu = True
                            print(f"found resource on cloud: {pilihan}")
                            break
                        elif ram_pilihan >= 0:
                            string1 = {
                                "module_name": str(row['Module']),
                                "app": str(applik),
                                "id_resource": pilihan
                            }
                            print(f"found resource on node {pilihan}")
                            node_spec.loc[int(pilihan), 'RAM'] = ram_pilihan
                            pilihan = random.choice(new_dict[pilihan])
                            print(f"next selection based on {pilihan}")
                            ketemu = True
                            break
                        else:
                            if row['Module'] == "Mod0":
                                pilihan = random.choice(new_dict[i['id_resource']])
                            else:
                                pilihan = random.choice(new_dict[pilihan])
                                print(f"current pilihan{pilihan} selection: {new_dict[pilihan]}")
                        count += 1
                    if ketemu:
                        print(f"we out from break inside {row['Module']}")
                        break
                else:
                    pilihan = random.choice(new_dict[pilihan])
                    print(f"failed to find node next to for {count}, finding new in{pilihan}")
                    break
                count_first += 1

            print("Count first : ", count_first)
            if count_first == 3 or pilihan == 100:
                print(f"this loop if fail to placing {count_first} or cloud because no placement {pilihan}")
                pilihan = 100
                string1 = {
                    "module_name": str(row['Module']),
                    "app": str(applik),
                    "id_resource": 100
                }


            allocation.append(string1)

            # # Store data in the DataFrame
            # allocation_df = allocation_df.append({
            #     'source_id': i['id_resource'],
            #     'module_name': row['Module'],
            #     'app_id': applik,
            #     'id_resource': string1['id_resource']
            # }, ignore_index=True)

    JSONfile["initialAllocation"] += allocation
    json_object = json.dumps(JSONfile, indent=2)

    with open(f"{save_folder}/{popul}_{app}_hop{hopnumber}.json", "w") as outfile:
        outfile.write(json_object)

    return True

# You can call the function like this:
# allocation_data = create_placement(data_pop, new_dict, app_spec, node_spec, popul, generate)
# allocation_data.to_csv("allocation_data.csv", index=False)  # Save the DataFrame to a CSV file


def txt_fail_noIPT(topology_file, population_file, application_file, save_folder, pop_id=0, app_id=0, hopnumber=2):
        """
        Run the placement algorithm with the given input files and parameters.
        
        Args:
            topology_file (str): Path to network topology JSON file
            population_file (str): Path to user population JSON file  
            application_file (str): Path to application specification JSON file
            save_folder (str): Folder to save output files
            pop_id (int): Population ID (default 0)
            app_id (int): Application ID (default 0)
            hopnumber (int): Number of hops (default 2)
        """
        import json

        # Load topology data
        with open(topology_file) as f:
            data = json.load(f)

        # Load population data
        with open(population_file) as f:
            data_pop = json.load(f)

        # Load application data
        with open(application_file) as f:
            data_app = json.load(f)

        # Process input data
        new_dict, source = create_dict(data)
        app_spec = app_specification(data_app)
        node_spec = node_specification(data)
        source = list(dict.fromkeys(source))

        # Run placement algorithm
        hopaware_txt_noIPT(data_pop, new_dict, app_spec, node_spec, 
                                 pop_id, app_id, save_folder, hopnumber=hopnumber)
        
        