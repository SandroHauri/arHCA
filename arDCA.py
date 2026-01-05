import numpy as np
import pickle as pkl
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import string
import MSA_helper
from sklearn.model_selection import train_test_split
import copy

from pytorch_models import arDCA
import time
from datetime import timedelta

model_params = {
                'order'         :   'entropy', #'entropy',#'ltr',
                'learn_rate'    :   0.0001,
                'l2_reg_dca'    :   2.e-6,
                'l2_reg_hoca'   :   1.e-5,
                'patience'      :   5,
                'max_epochs'    :   1000,
                'batch_size'    :   1024,
                'use_gpu'       :   True,
                'n_layers'      :   [2],
                'n_hiddens'     :   [64]
                }

order = model_params['order']


datasets = ['DLG4_RAT', 'PABP_YEAST'] # # #'PYP_HALHA'
data_dir = 'Marks_data'

for dataset in datasets:
    print(dataset)
    try:
        data = pkl.load(open(data_dir + f'/data_nogaps_{dataset}.pkl', 'rb'))
    except:
        data = MSA_helper.DataHelper(dataset, theta=0.2, calc_weights=True, working_dir=data_dir)
        pkl.dump(data, open(data_dir + f'/data_nogaps_{dataset}.pkl', 'wb'))
    
    
    s_name = f'models/test_nogaps_{dataset}_{order}'
    
    x_train, x_val, w_train, w_val = train_test_split(data.x_train, data.weights, test_size=0.2, random_state=1337)
    print("x_train shape", x_train.shape)
    print("x_val shape", x_val.shape)
    print("w_train shape", w_train.shape)
    print("w_val shape", w_val.shape)
    
    alphabet_len = len(data.alphabet)
    
    if model_params['use_gpu'] and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(device)
    
    def ce_weighted_per_pos(pred, label, weights):
        (N, L, W) = pred.shape
        label = label.reshape((N, L, W))
        
        log_probs = torch.log(pred + 1e-9)
        inv_log_probs = torch.log(1 - pred + 1e-9)
        
        cross_entropy = - label*log_probs - (1-label)*inv_log_probs
        entropy = cross_entropy.sum(-1)
        
        loss = (entropy * weights.unsqueeze(1)).mean(0)
        return loss
    
    seq_len = x_train.shape[1]
    
    N_train = x_train.shape[0]
    x_tmp = (x_train * w_train[..., np.newaxis, np.newaxis])
    x_tmp = x_tmp.reshape( (-1, seq_len, alphabet_len) ).sum(0)
    indep_model = x_tmp / w_train.sum()
    
    
    # %% Train
    print('start training')
    if model_params['order'] == 'ltr':
        pos_seq = list(range(seq_len))
    elif model_params['order'] == 'rtl':
        pos_seq = list(range(seq_len-1,-1,-1))
    elif model_params['order'] == 'entropy':
        entropy_per_pos = (- indep_model * np.log(indep_model+1e-9)).sum(-1)
        pos_seq = entropy_per_pos.argsort().tolist()
    elif model_params['order'] == 'r_entropy':
        entropy_per_pos = (- indep_model * np.log(indep_model+1e-9)).sum(-1)
        pos_seq = entropy_per_pos.argsort().tolist()[::-1]
    elif model_params['order'] == 'random':
        np.random.seed(987)
        pos_seq = np.random.choice(seq_len, seq_len, replace=False)
    elif model_params['order'] == 'r_random':
        np.random.seed(987)
        pos_seq = np.random.choice(seq_len, seq_len, replace=False)
        pos_seq = pos_seq[::-1]
    # it's a MSA with added synthetic positions

    
    x_train = x_train.reshape( (N_train, seq_len, alphabet_len) )
    x_train = x_train[:, pos_seq]
    x_val = x_val.reshape( (-1, seq_len, alphabet_len) )
    x_val = x_val[:, pos_seq]
    
    patience =  model_params['patience']
    batch_size = model_params['batch_size']
    learn_rate = model_params['learn_rate']
        
    starttime = time.time()
    sample_order = np.arange(N_train)
    
    
    train_loss_list = []; hist = []
    starttime = time.time()
    for layers in model_params['n_layers']:
        for hidden in model_params['n_hiddens']:
            s_name_plus = s_name + f'_dca'            
            
            model = arDCA(seq_len = seq_len, l_alphabet = alphabet_len).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=model_params['l2_reg_dca'])
            
            
            epoch = 0; n_processed = 0; helper_count = 0; best_epoch = 0
            best_loss = np.array([1e9]*seq_len) # 1e9  # np.array([1e9]*seq_len)
            while epoch < model_params['max_epochs']:
                batch_index = np.random.choice(sample_order, batch_size).tolist()
                X = x_train[batch_index]
                weights = torch.tensor(w_train[batch_index]).to(device)
                inp_data = torch.Tensor(X).to(device)
                target_data = torch.Tensor(X).to(device).reshape( (batch_size, -1) )
                
                y_hca = model.probs(inp_data)
                tot_loss_per_pos = ce_weighted_per_pos(y_hca, target_data, weights)
                tot_loss = tot_loss_per_pos.mean()
            
            
                optimizer.zero_grad()
                tot_loss.backward()
                optimizer.step()
                
                train_loss_list.append(tot_loss_per_pos.detach().cpu().numpy())
                    
                n_processed += batch_size
                epoch = n_processed // N_train
                
                if int(epoch) > helper_count:
                    n_val = x_val.shape[0]
                    val_loss_list = []
                    for b in range(0, n_val, batch_size):
                        X = x_val[b:b+batch_size]
                        weights = torch.Tensor(w_val[b:b+batch_size]).to(device)
                        inp_data = torch.Tensor(X).to(device)
                        target_data = torch.Tensor(X).to(device).reshape( (X.shape[0], -1) )
                        
                        y_hca = model.probs(inp_data)
                        tot_loss_per_pos = ce_weighted_per_pos(y_hca, target_data, weights)
                        val_loss_list.append(tot_loss_per_pos.detach().cpu().numpy())
            
                    helper_count = int(epoch)
                    
                    train_loss = np.array(train_loss_list).mean(0)
                    train_loss_list = []
                    val_loss = np.array(val_loss_list).mean(0)
                    hist.append((train_loss, val_loss))
                    
                    '''
                    if val_loss.mean() < best_loss:
                        best_loss = val_loss.mean()
                        best_epoch = epoch
                        torch.save(model, s_name_plus + '.pt')
                    '''    
                    
                    if (val_loss < best_loss).any():
                        best_model = copy.deepcopy(model).to('cpu')
                        for i in range(seq_len):
                            if val_loss[i] < best_loss[i]:
                                best_loss[i] = val_loss[i]
                                if i > 0:
                                    best_model.layers[i-1] = copy.deepcopy(model.layers[i-1])
                                else:
                                    best_model.nn_bias0 = copy.deepcopy(model.nn_bias0)
                        best_epoch = epoch
                        torch.save(best_model, s_name_plus + '.pt')
                    
                    
                    if epoch - best_epoch > patience:
                        break
                        
                    if len(hist)>0 and len(hist)%10 == 0:
                        print(f'epoch {helper_count}    elapsed time: {timedelta(seconds=time.time()-starttime)}')
            del model
    
    print("done")
    print(f'elapsed time: {timedelta(seconds=time.time()-starttime)}')
    
