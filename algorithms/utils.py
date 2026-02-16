import csv
import ndjson
import numpy as np
import matplotlib.pyplot as plt


def read_list_of_lists_from_csv(loc):
    all_lists = []
    with open(loc, 'r', newline='') as f:
        csvFile = csv.reader(f)
        for lines in csvFile:
            all_lists.append(lines)

    return all_lists


def find_env_ids(env, json_loc):
    # load from file-like objects
    with open(json_loc) as f:
        all_data = ndjson.load(f)

    matches = [row['id'] for row in all_data 
            if row['env'] == env
        ]

    return matches


def display_id_results(ids):

    for id in ids:
        loc = f'results/rewards_lists/{id}.csv'

        
        lists = read_list_of_lists_from_csv(loc)

        rewards_ar = np.array(lists).astype(np.int_)
        median = np.quantile(rewards_ar, 0.5, axis = 0)
        plt.plot(median[0:500], label = id)
        print(id)
    plt.legend()
    plt.show()

def fetch_id_params(ids, loc):

    with open(loc) as f:
        all_data = ndjson.load(f)

    rel_data = []
    for id in ids:
        # load from file-like objects

        matches = next((row for row in all_data 
                if row['id'] == id
            ), None)

        rel_data.append(matches)

    return rel_data
