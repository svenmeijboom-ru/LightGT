import os
import torch
import torch.nn as nn
import numpy as np
import random
from time import time
from tqdm import tqdm
from dataloader import EvalDataset, data_load, TrainingDataset
from model import Net
from torch.utils.data import DataLoader
from Parser import parse_args
import sparsity_group_test as sgt


if __name__ == '__main__':

    args = parse_args()

    start_time = time()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    dataset = args.dataset
    
    save_file = 0
    while True:
        if os.path.exists('./Result/'+dataset+'/result_{0}.txt'.format(save_file)):
            save_file += 1
        else:
            path = './Result/'+dataset+'/result_{0}.txt'.format(save_file)
            break
    final_path = './Result/'+dataset+'/result.txt'

    learning_rate = args.l_r
    weight_decay = args.weight_decay
    batch_size = args.batch_size 
    num_workers = args.num_workers
    num_epoch = args.num_epoch
    topK = args.topK
    prefix = args.prefix
    aggr_mode = args.aggr_mode

    has_v = False#True if args.has_v == 'True' else False
    has_a = False#True if args.has_a == 'True' else False
    has_t = False#True if args.has_t == 'True' else False
    has_entropy_loss = True if args.has_entropy_loss == 'True' else False
    has_weight_loss = True if args.has_weight_loss == 'True' else False
    dim_E = args.dim_E
    src_len = args.src_len
    transformer_layers = args.transformer_layers
    nhead = args.nhead
    lightgcn_layers = args.lightgcn_layers
    score_weight = args.score_weight

    with open(path, 'a') as save_file:
        save_file.write('lr: {0} Weight_decay:{1} Batsh_size:{2} src_len:{3} Transformer_layers:{4} nhead:{5} LightGCN_layers:{6} Score_weight:{7}\n\n'.
                            format(learning_rate, weight_decay, batch_size, src_len, transformer_layers, nhead, lightgcn_layers, score_weight))

    print('lr: {0} Weight_decay:{1} Batsh_size:{2} src_len:{3} Transformer_layers:{4} nhead:{5} LightGCN_layers:{6}'.
                        format(learning_rate, weight_decay, batch_size, src_len, transformer_layers, nhead, lightgcn_layers))
    # exit()
    #####################################################################################################
    
    print('Data loading ...')

    user_num, item_num, train_edge, user_item_dict, v_feat, a_feat, t_feat = data_load(dataset, has_v, has_a, has_t)

    train_dataset = TrainingDataset(dataset, user_num, item_num, user_item_dict, train_edge, src_len=src_len)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    graph = train_dataset.get_sparse_graph()
    eval_dataset = EvalDataset(dataset, user_num, item_num, user_item_dict, src_len)
    eval_dataloader = DataLoader(eval_dataset, 2000, shuffle=False, num_workers=num_workers)

    val_data = np.load('./Data/'+dataset+'/val.npy', allow_pickle=True)
    val_data = [[*[i[0]], *[j+user_num for j in i[1:]]] for i in val_data]
    test_data = np.load('./Data/'+dataset+'/test.npy', allow_pickle=True)
    test_data = [[*[i[0]], *[j+user_num for j in i[1:]]] for i in test_data]
    
    print('Data has been loaded.')

    # conf = {
    #     "step": 5,
    #     "batch_size_test": 2048,
    #     "device": "cuda:0",
    #     "src_len": src_len,
    #     "user_num": user_num,
    #     "item_num": item_num,
    #     "topk": [10],
    # }
    # data = {
    #     "graph": np.load('./Data/'+dataset+'/test.npy', allow_pickle=True),
    #     "graph_train": user_item_dict, # graph: sp_graph, [#users, #bundles]
    # }


    #####################################################################################################

    model = Net(user_num, item_num, graph, user_item_dict, v_feat, a_feat, t_feat, eval_dataloader, weight_decay, 
                src_len, batch_size, dim_E, transformer_layers, nhead, lightgcn_layers, score_weight).cuda()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    #####################################################################################################

    #max_precision = 0.0
    max_recall_expl = max_recall_rep = 0.0
    max_phr_expl = max_phr_rep = 0.0
    #max_NDCG = 0.0
    val_max_recall_expl = val_max_recall_rep = 0.0
    num_decreases = 0
    test_max_recall_expl = test_max_recall_rep = 0.0
    test_max_phr_expl = test_max_phr_rep = 0.0
    #max_hit_ratio = 0.0

    #####################################################################################################

    for epoch in range(num_epoch):
        epoch_start = time()
        model.train()
        print('Training start .. ')
        sum_loss = 0.0
        sum_model_loss = 0.0
        sum_reg_loss = 0.0
        sum_ent_loss = 0.0
        sum_weight_loss = 0.0
        step = 0.0
        pbar = tqdm(total=len(train_dataset))
        for users, items, user_item, mask in train_dataloader:
            optimizer.zero_grad()
            loss, model_loss, reg_loss, weight_loss, entropy_loss = model.loss(users, items, user_item, mask)
            loss.backward(retain_graph=True)
            optimizer.step()
            
            pbar.update(batch_size)
            sum_loss += loss.cpu().item()
            sum_model_loss += model_loss.cpu().item()
            sum_reg_loss += reg_loss.cpu().item()
            sum_ent_loss += entropy_loss.cpu().item()
            sum_weight_loss += weight_loss.cpu().item()
            step += 1.0
        
        pbar.close()
        print('--------- loss value:{}  model_loss value:{}  reg_loss value:{} entropy_loss value:{} weight_loss value:{}--------------'
            .format(sum_loss/step, sum_model_loss/step, sum_reg_loss/step, sum_ent_loss/step, sum_weight_loss/step))
        with open(path, 'a') as save_file:
            save_file.write('--------- loss value:{}  model_loss value:{}  reg_loss value:{} entropy_loss value:{} weight_loss value:{}--------------\n\n'
            .format(sum_loss/step, sum_model_loss/step, sum_reg_loss/step, sum_ent_loss/step, sum_weight_loss/step))

        if torch.isnan(loss):
            with open(path, 'a') as save_file:
                    save_file.write('lr: {0} \t Weight_decay:{1} is Nan\r\n'.format(learning_rate, weight_decay))
            break

        torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            epoch_time = time() - epoch_start
            h = int(epoch_time // 3600)
            m = int((epoch_time % 3600) // 60)
            s = int(epoch_time % 60)

            print('Val start ...')
            val_recall_expl, val_recall_rep, val_phr_expl, val_phr_rep, explr, repr = model.full_accuracy(val_data, user_item_dict)
            torch.cuda.empty_cache()
            print('------------------{0}-th RecallExpl:{1:.4f} RecallRep:{2:.4f} PHRExpl:{3:.4f} PHRRep:{4:.4f} ExplR:{5:.4f} Repr:{6:.4f} Epoch_time:{7}:{8}:{9}---------------------------------'.format(
                epoch, val_recall_expl, val_recall_rep, val_phr_expl, val_phr_rep, explr, repr, h, m, s))
                
            print('Test start ...')
            test_recall_expl, test_recall_rep, test_phr_expl, test_phr_rep, expl, repr = model.full_accuracy(test_data, user_item_dict)
            torch.cuda.empty_cache()

            print('------------------{0}-th RecallExpl:{1:.4f} RecallRep:{2:.4f} PHRExpl:{3:.4f} PHRRep:{4:.4f} ExplR:{5:.4f} Repr:{6:.4f} Epoch_time:{7}:{8}:{9}---------------------------------'.format(
                epoch, test_recall_expl, test_recall_rep, test_phr_expl, test_phr_rep, explr, repr, h, m, s))

            with open(path, 'a') as save_file:
                save_file.write('------------------{0}-th RecallExpl:{1:.4f} RecallRep:{2:.4f} PHRExpl:{3:.4f} PHRRep:{4:.4f} Epoch_time:{5}:{6}:{7}---------------------------------\n'.format(
                epoch, val_recall_expl, val_recall_rep, val_phr_expl, val_phr_rep, h, m, s))
                save_file.write('------------------{0}-th RecallExpl:{1:.4f} RecallRep:{2:.4f} PHRExpl:{3:.4f} PHRRep:{4:.4f} Epoch_time:{5}:{6}:{7}---------------------------------\n\n'.format(
                epoch, test_recall_expl, test_recall_rep, test_phr_expl, test_phr_rep, h, m, s))

            # groups = sgt.sparsity_group_test(model, conf, data)
            # sgt.print_group_metrics(groups)

        torch.cuda.empty_cache()

        if val_recall_expl > val_max_recall_expl:
            val_max_recall_expl = val_recall_expl
            #max_precision = test_precision
            max_recall_expl = test_recall_expl
            #max_NDCG = test_ndcg
            max_phr_expl = test_phr_expl
            num_decreases = 0
            #max_hit_ratio = val_hit_ratio
        if val_recall_rep > val_max_recall_rep:
            val_max_recall_rep = val_recall_rep
            #max_precision = test_precision
            max_recall_rep = test_recall_rep
            #max_NDCG = test_ndcg
            max_phr_rep = test_phr_rep
            num_decreases = 0
            #max_hit_ratio = val_hit_ratio
        else:
            if num_decreases > 20:
                with open(path, 'a') as save_file:
                    save_file.write('lr: {0} \t Weight_decay:{1} =====> RecalExpl:{2} \t PHRExpl:{3} \t RecallRep:{4} \t PHRRep:{5}\r\n'.
                                    format(learning_rate, weight_decay, max_recall_expl, max_phr_expl, max_recall_rep, max_phr_rep))

                    total_time = time()-start_time
                    h = int(total_time // 3600)
                    m = int((total_time % 3600) // 60)
                    s = int(total_time % 60)
                    save_file.write('time:  {0}:{1}:{2}\n\n'.format(h, m, s))

                with open(final_path, 'a') as save_file:
                    save_file.write('lr: {0} Weight_decay:{1} Batsh_size:{2} src_len:{3} Transformer_layers:{4} nhead:{5} LightGCN_layers:{6} Score_weight:{7}\n'.
                                    format(learning_rate, weight_decay, batch_size, src_len, transformer_layers, nhead, lightgcn_layers, score_weight))
                    save_file.write('RecalExpl:{0} \t PHRExpl:{1} \t RecallRep:{2} \t PHRRep:{3}\n\n'.
                                    format(max_recall_expl, max_phr_expl, max_recall_rep, max_phr_rep))

                break
            else:
                num_decreases += 1