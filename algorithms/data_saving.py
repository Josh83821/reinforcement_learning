import ndjson
import csv
from uuid import uuid4
import numpy as np


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(v) for v in obj)
    elif isinstance(obj, np.generic):
        return obj.item()  # converts np.int64 → int, np.float64 → float
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def write_list_to_csv(loc, data):
    with open(loc, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)      # Write data rows


def read_list_of_lists_from_csv(loc):
    all_lists = []
    with open(loc, 'r', newline='') as f:
        csvFile = csv.reader(f)
        for lines in csvFile:
            all_lists.append(lines)

    return all_lists

    
def check_for_id_already(model_data_loc, alg_data):
    
    # load from file-like objects
    with open(model_data_loc) as f:
        all_data = ndjson.load(f)
    
    match = next(
        (row for row in all_data 
         if row['model_opt'] == alg_data['model_opt'] 
         and row['hypers'] == alg_data['hypers']
         and row['env'] == alg_data['env']
         and row['alg_name'] == alg_data['alg_name']),
        None
    )

    if match:
        return match['id']
    else:
        return None


def save_new_alg(loc, data):
    id = str(uuid4())
    data['id'] = id
    with open(loc, "a") as f:
        f.write(ndjson.dumps([data]) + "\n")
    return id

def save_data(loss_lists, rewards_lists, alg_data):
    
    loss_lists = make_json_serializable(loss_lists)
    rewards_lists = make_json_serializable(rewards_lists)
    alg_data = make_json_serializable(alg_data)

    alg_data_loc = "results/model_alg_data.ndjson"

    id = check_for_id_already(alg_data_loc, alg_data)
    if not id:
        print(alg_data)
        id = save_new_alg(alg_data_loc, alg_data)

    rewards_loc = f"results/rewards_lists/{id}.csv"
    loss_loc = f"results/loss_lists/{id}.csv"


    for rewards_list in rewards_lists:
        write_list_to_csv(rewards_loc, rewards_list)
    for loss_list in loss_lists:
        write_list_to_csv(loss_loc, loss_list)
