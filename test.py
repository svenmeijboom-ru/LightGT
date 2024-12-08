import numpy as np
import json

dataset = 'instacart/'
dir_str = './Data/' + dataset

# Open and read the JSON file
with open(dir_str+'instacart_merged.json', 'r') as file:
    user_item_dict = json.load(file)

train_edge = []
val = []
test = []
for k in user_item_dict.keys():
    v = user_item_dict[k].pop()
    t = user_item_dict[k].pop()
    
    val.append([int(k)] + v)
    test.append([int(k)] + t)

    for i in user_item_dict[k]:
        for ii in i:
            train_edge.append([int(k), ii])

#print(val)
val_array_with_lists = np.array(val, dtype=object)
test_array_with_lists = np.array(test, dtype=object)

#print(user_item_dict)
np.save(dir_str+'train.npy', train_edge)
np.save(dir_str+'val.npy', val_array_with_lists)
np.save(dir_str+'test.npy', test_array_with_lists)