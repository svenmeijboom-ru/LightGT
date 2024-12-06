import numpy as np
import pandas as pd

train_edge = np.load('./Data/movielens/train.npy', allow_pickle=True)
user_item_dict = np.load('./Data/movielens/user_item_dict.npy', allow_pickle=True)
v_feat = np.load('./Data/movielens/FeatureVideo_normal.npy', allow_pickle=True)
a_feat = np.load('./Data/movielens/FeatureAudio_avg_normal.npy', allow_pickle=True)
t_feat = np.load('./Data/movielens/FeatureText_stl_normal.npy', allow_pickle=True)


df_train_edge = pd.DataFrame(train_edge)
#df_user_item_dict = pd.DataFrame.from_dict(user_item_dict)
df_v_feat = pd.DataFrame(v_feat)
df_a_feat = pd.DataFrame(a_feat)
df_t_feat = pd.DataFrame(t_feat)

df_train_edge.to_excel('train_edge_xlsx.xlsx', index=False)
#df_user_item_dict.to_excel('user_item_dict_xlsx.xlsx', index=False)
df_v_feat.to_excel('v_feat_xlsx.xlsx', index=False)
df_a_feat.to_excel('a_feat_xlsx.xlsx', index=False)
df_t_feat.to_excel('t_feat_xlsx.xlsx', index=False)