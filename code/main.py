import os
from random import *
import time

import input_data
import model_torch as model

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import tool
import math

from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

a = Random()
a.seed(1)

def setting(data):
    vocab_size, word_emb_size = data['embedding'].shape
    sample_num, max_time = data['x_tr'].shape
    test_num = data['x_te'].shape[0]
    s_cnum = np.unique(data['y_tr']).shape[0]
    u_cnum = np.unique(data['y_te']).shape[0]
    config = {}
    config['keep_prob'] = 0.8 # embedding dropout keep rate
    config['hidden_size'] = 32 # embedding vector size
    config['batch_size'] = 64 # vocab size of word vectors
    config['vocab_size'] = vocab_size # vocab size of word vectors (10,895)
    config['num_epochs'] = 200 # number of epochs
    config['max_time'] = max_time
    config['sample_num'] = sample_num #sample number of training data
    config['test_num'] = test_num #number of test data
    config['s_cnum'] = s_cnum # seen class num
    config['u_cnum'] = u_cnum #unseen class num
    config['word_emb_size'] = word_emb_size # embedding size of word vectors (300)
    config['d_a'] = 20 # self-attention weight hidden units number
    config['output_atoms'] = 10 #capsule output atoms
    config['r'] = 3 #self-attention weight hops
    config['num_routing'] = 2 #capsule routing num
    config['alpha'] = 0.0001 # coefficient of self-attention loss
    config['margin'] = 1.0 # ranking loss margin
    config['learning_rate'] = 0.0001
    config['sim_scale'] = 4 #sim scale
    config['nlayers'] = 2 # default for bilstm
    config['ckpt_dir'] = './saved_models/' #check point dir

    return config

def get_sim(data):
    # get unseen and seen categories similarity
    s = normalize(data['sc_vec'])
    u = normalize(data['uc_vec'])
    sim = tool.compute_label_sim(u, s, config['sim_scale'])
    return sim

def evaluate_test(data, config, lstm,embedding):
    # zero-shot testing state
    # seen votes shape (110, 2, 34, 10)
    x_te = data['x_te']
    y_te_id = data['y_te']
    u_len = data['u_len']
    y_ind = data['s_label']
    # get unseen and seen categories similarity
    # sim shape (8, 34)
    sim_ori = torch.from_numpy(get_sim(data))
    total_unseen_pred = np.array([], dtype=np.int64)
    total_y_test = np.array([], dtype=np.int64)
    batch_size  = config['test_num']
    test_batch = int(math.ceil(config['test_num'] / float(batch_size)))
    with torch.no_grad():
        for i in range(test_batch):
            begin_index = i * batch_size
            end_index = min((i + 1) * batch_size, config['test_num'])
            batch_te_original = x_te[begin_index : end_index]
            batch_len = u_len[begin_index : end_index]
            batch_test = y_te_id[begin_index: end_index]
            batch_len = torch.from_numpy(batch_len)

            # sort by descending order for pack_padded_sequence
            batch_len, perm_idx = batch_len.sort(0, descending=True)
            batch_te = batch_te_original[perm_idx]
            batch_test = batch_test[perm_idx]
            batch_te = torch.from_numpy(batch_te)

            lstm(batch_te, batch_len, embedding)
            attentions, seen_logits, seen_votes, seen_weights_c = lstm.attention, lstm.logits, \
                                                                  lstm.votes, lstm.weights_c
            sim = np.expand_dims(sim_ori,0)
            sim =  np.tile(sim, [seen_votes.shape[1],1,1])
            sim = np.expand_dims(sim, 0)
            sim = np.tile(sim, [seen_votes.shape[0],1,1,1])
            seen_weights_c = np.tile(np.expand_dims(seen_weights_c, -1), [1,1,1, config['output_atoms']])
            mul = np.multiply(seen_votes, seen_weights_c)

            # compute unseen features
            # unseen votes shape (110, 2, 8, 10)
            unseen_votes = np.matmul(sim, mul)

            # routing unseen classes
            torch_unseen_votes = torch.from_numpy(unseen_votes)
            u_activations, u_weights_c = update_unseen_routing(torch_unseen_votes, config, 3)
            unseen_logits = torch.norm(u_activations, dim=-1)
            te_logits = unseen_logits
            te_batch_pred = np.argmax(te_logits, 1)
            total_unseen_pred = np.concatenate((total_unseen_pred, te_batch_pred))
            total_y_test = np.concatenate((total_y_test, batch_test))
            print ("           zero-shot intent detection test set performance        ")
            acc = accuracy_score(total_y_test, total_unseen_pred)
            print (classification_report(total_y_test, total_unseen_pred, digits=4))
    return acc

def generate_batch(n, batch_size):
    batch_index = a.sample(xrange(n), batch_size)
    return batch_index

def _squash(input_tensor):
    norm = torch.norm(input_tensor, dim=2, keepdim=True)
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (0.5 + norm_squared))


def update_unseen_routing(votes, config, num_routing=3):
    votes_t_shape = [3, 0, 1, 2]
    r_t_shape = [1, 2, 3, 0]
    votes_trans = votes.permute(votes_t_shape)
    num_dims = 4
    input_dim = config['r']
    output_dim = config['u_cnum']
    input_shape = votes.shape
    logit_shape = np.stack([input_shape[0], input_dim, output_dim])
    logits = torch.zeros(logit_shape[0], logit_shape[1], logit_shape[2])
    activations = []


    for iteration in range(num_routing):
        route = F.softmax(logits, dim=2)
        preactivate_unrolled = route * votes_trans
        preact_trans = preactivate_unrolled.permute(r_t_shape)

        # delete bias to fit for unseen classes
        preactivate = torch.sum(preact_trans, dim=1)
        activation = _squash(preactivate)
        # activations = activations.write(i, activation)
        activations.append(activation)
        # distances: [batch, input_dim, output_dim]
        act_3d = torch.unsqueeze(activation, 1)
        tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
        tile_shape[1] = input_dim
        act_replicated = act_3d.repeat(tile_shape)
        distances = torch.sum(votes * act_replicated, dim=3)
        logits = logits + distances

    return activations[num_routing-1], route

def sort_batch(batch_x, batch_y, batch_len, batch_ind):
    batch_len_new = torch.from_numpy(batch_len)
    batch_len_new, perm_idx = batch_len_new.sort(0, descending=True)
    batch_x_new = batch_x[perm_idx]
    batch_y_new = batch_y[perm_idx]
    batch_ind_new = batch_ind[perm_idx]

    return torch.from_numpy(batch_x_new), torch.from_numpy(batch_y_new), \
           batch_len_new, torch.from_numpy(batch_ind_new)

if __name__ == "__main__":
    # load data
    data = input_data.read_datasets()
    x_tr = data['x_tr']
    y_tr = data['y_tr']
    y_tr_id = data['y_tr']
    y_te_id = data['y_te']
    y_ind = data['s_label']
    s_len = data['s_len']
    embedding = data['embedding']

    x_te = data['x_te']
    u_len = data['u_len']
    # load settings
    config = setting(data)

    # Training cycle
    batch_num = config['sample_num'] / config['batch_size']
    overall_train_time = 0.0
    overall_test_time = 0.0

    lstm = model.CapsuleNetwork(config)
    optimizer = optim.Adam(lstm.parameters(), lr=config['learning_rate'])

    if os.path.exists(config['ckpt_dir'] + 'best_model.pth'):
        print("Restoring weights from previously trained rnn model.")
        lstm.load_state_dict(torch.load(config['ckpt_dir'] + 'best_model.pth' ))
    else:
        print('Initializing Variables')
        os.mkdir(config['ckpt_dir'])

    best_acc = 0

    for epoch in range(config['num_epochs']):
        lstm.train()
        avg_acc = 0.0;
        epoch_time = time.time()
        for batch in range(batch_num):

            batch_index = generate_batch(config['sample_num'], config['batch_size'])

            batch_x = x_tr[batch_index]
            batch_y_id = y_tr_id[batch_index]
            batch_len = s_len[batch_index]
            batch_ind = y_ind[batch_index]

            # sort by descending order for pack_padded_sequence
            batch_x, batch_y_id, batch_len, batch_ind = sort_batch(batch_x, batch_y_id, batch_len, batch_ind)

            output = lstm.forward(batch_x, batch_len,torch.from_numpy(embedding))
            loss_val = lstm.loss(batch_ind)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            clone_logits = lstm.logits.detach().clone()
            tr_batch_pred = np.argmax(clone_logits, 1)
            acc = accuracy_score(batch_y_id, tr_batch_pred)
            avg_acc += acc

        train_time = time.time() - epoch_time
        overall_train_time += train_time
        print "------epoch : ", epoch, " Loss: ", loss_val.item(), " Acc:", round((avg_acc / batch_num), 4), " Train time: ", \
                                round(train_time, 4), "--------"

        lstm.eval()
        cur_acc = evaluate_test(data, config, lstm,torch.from_numpy(embedding))
        if cur_acc > best_acc:
            # save model
            best_acc = cur_acc
            torch.save(lstm.state_dict(), config['ckpt_dir'] + 'best_model.pth')

        print("cur_acc", cur_acc)
        print("best_acc", best_acc)
        test_time = time.time() - epoch_time
        overall_test_time += test_time
        print("Testing time", round(test_time, 4))

    print("Overall training time", overall_train_time)
    print("Overall testing time", overall_test_time)

