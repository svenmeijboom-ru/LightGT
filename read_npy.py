import numpy as np
import pandas as pd
import itertools


train_edge = np.load('./Data/instacart/train.npy', allow_pickle=True)
print(train_edge)
print(len(train_edge))
#user_item_dict = np.load('./Data/movielens/train_edge.npy', allow_pickle=True).item()
#for i in user_item_dict.keys():
#    user_item_dict[i] = list(itertools.chain.from_iterable(user_item_dict[i]))
#print(len(user_item_dict))
#user_item_dict = dict(enumerate(user_item_dict.flatten()))[0]
#print(len(user_item_dict))

#settotal = []
#settotal = set(settotal)
#for user in user_item_dict:
#    #set(list(user_item_dict[user]))
#    settotal.update(set(user_item_dict[user]))
#print(len(settotal))
#print(max(settotal))

#v_feat = np.load('./Data/movielens/FeatureVideo_normal.npy', allow_pickle=True)
#a_feat = np.load('./Data/movielens/FeatureAudio_avg_normal.npy', allow_pickle=True)
#t_feat = np.load('./Data/movielens/FeatureText_stl_normal.npy', allow_pickle=True)
#val = np.load('./Data/instacart/val.npy', allow_pickle=True)
#test = np.load('./Data/instacart/test.npy', allow_pickle=True)

#d = dict(enumerate(user_item_dict.flatten(), 1))
#print(d[1]['1'])#[2533])
#print(user_item_dict)
#user_item_dict = dict(enumerate(user_item_dict.flatten()))[0]
#print(user_item_dict)
#print(user_item_dict[0])
#print(user_item_dict[54])



#df_train_edge = pd.DataFrame(train_edge)
#df_user_item_dict = pd.DataFrame(user_item_dict)
#df_v_feat = pd.DataFrame(v_feat)
#df_a_feat = pd.DataFrame(a_feat)
#df_t_feat = pd.DataFrame(t_feat)
#df_val = pd.DataFrame(val)
#df_test = pd.DataFrame(test)

#print(len(val))
#print(len(test))

# df_train_edge.to_excel('train_edge_xlsx.xlsx', index=False)
#df_user_item_dict.to_excel('user_item_dict_xlsx_instacart.xlsx', index=False)
# df_v_feat.to_excel('v_feat_xlsx.xlsx', index=False)
# df_a_feat.to_excel('a_feat_xlsx.xlsx', index=False)
# df_t_feat.to_excel('t_feat_xlsx.xlsx', index=False)
#df_val.to_excel('val_xlsx.xlsx', index=False)
#df_test.to_excel('test_xlsx.xlsx', index=False)