import numpy as np
import json

dataset = 'instacart/'
dir_str = './Data/' + dataset

# Open and read the JSON file
with open(dir_str+'instacart_merged.json', 'r') as file:
    user_item_dict = json.load(file)

train_edge = []
for k in user_item_dict.keys():
    for i in user_item_dict[k]:
        for ii in i:
            train_edge.append([int(k), ii])
    break


np.save(dir_str+'train.npy', train_edge)
np.save(dir_str+'user_item_dict.npy', user_item_dict)