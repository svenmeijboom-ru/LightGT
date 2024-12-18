import numpy as np
import json

dataset = 'dunnhumby/'
dir_str = './Data/' + dataset

# Open and read the JSON file
with open(dir_str+'dunnhumby_merged.json', 'r') as file:
    old_user_item_dict = json.load(file)

train_edge = []
val = []
test = []

user_item_dict = {int(k): v for k, v in old_user_item_dict.items()}


for k in user_item_dict.keys():
    if len(user_item_dict[k]) > 2:
        v = user_item_dict[k].pop()
        t = user_item_dict[k].pop()
    else:
        v = []
        t = []
    
    val.append([int(k)] + v)
    test.append([int(k)] + t)

    for i in user_item_dict[k]:
        for ii in i:
            train_edge.append([int(k)-1, ii])

#print(val)
val_array_with_lists = np.array(val, dtype=object)
test_array_with_lists = np.array(test, dtype=object)

#print(user_item_dict)
np.save(dir_str+'user_item_dict.npy', user_item_dict)
np.save(dir_str+'train.npy', train_edge)
np.save(dir_str+'val.npy', val_array_with_lists)
np.save(dir_str+'test.npy', test_array_with_lists)