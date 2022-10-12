from azureml.core import Workspace

# Creating a Workspace
ws = Workspace(name = "aml-workspace", subsrciption_id = "12d2e1d21-e12d1221-2asf1s2", resource_group = "aml-resources", 
               create_resource_group = True, location = "eastus")
               

# Connecting to a Workspace

    # By Config.json file
ws = Workspace.from_config() # Default Path is python file path.

    # By using .get()
ws = Workspace.get(name = "aml-workspace", subsrciption_id = "12d2e1d21-e12d1221-2asf1s2", resource_group = "aml-resources")

# Seeing all the Compute resources we have in our workspace
for compute_name in ws.compute_targets:
    compute = ws.compute_targets[compute_name]
    print(compute.name + ":" + compute.type)
   
# Seeing all the Datastores and Datasets in our workspace
from azurecore.ml import Datastore, Datasets

for datastore_name in ws.datasources:
    datastore = Datastore.get(ws, datastore_name)
    print(datastore.name, ":", datastore.datastore_type)
   
for dataset_name in list(ws.datasets.keys()):
    dataset = Dataset.get_by_name(ws, dataset_name)
    print(dataset.name)

