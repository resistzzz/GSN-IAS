
import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import time


def transfer2idx(data, item2idx):
    seqs, labs = data[0], data[1]
    for i in range(len(seqs)):
        data[0][i] = [item2idx[s] for s in data[0][i]]
        data[1][i] = item2idx[data[1][i]]
    return data


def handle_adj(adj_items, weight_items, n_items, sample_num):
    adj_entity = np.zeros((n_items, sample_num), dtype=np.int)
    wei_entity = np.zeros((n_items, sample_num))
    for entity in range(1, n_items):
        neighbor = list(adj_items[entity])
        neighbor_weight = list(weight_items[entity])
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            continue
        if n_neighbor >= sample_num:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
        tmp, tmp_wei = [], []
        for i in sampled_indices:
            tmp.append(neighbor[i])
            tmp_wei.append(neighbor_weight[i])

        adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
        wei_entity[entity] = np.array([neighbor_weight[i] for i in sampled_indices])
    return adj_entity, wei_entity


def handle_adj_with_pop(adj_items, weight_items, n_items, anchors, prob, sample_num):
    adj_entity = np.zeros((n_items, sample_num), dtype=np.int)
    # wei_entity = np.zeros((n_items, sample_num))
    for entity in range(1, n_items):
        neighbor = list(adj_items[entity])
        neighbor_weight = list(weight_items[entity])
        neighbor_weight = np.array(neighbor_weight) / np.array(neighbor_weight).sum()
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            continue
        if n_neighbor >= sample_num:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False, p=neighbor_weight)
            adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
        else:
            entity_prob = prob[entity]
            anchor_index = np.random.choice(list(range(len(anchors))), size=sample_num - n_neighbor, replace=True, p=entity_prob)
            sampled_anchor = np.array(anchors)[anchor_index]
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=n_neighbor, replace=False, p=neighbor_weight)
            adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices] + sampled_anchor.tolist())

    return adj_entity


class Data(object):
    '''Data用每个batch中的最长长度来进行padding'''
    def __init__(self, data, n_items):
        self.data = data
        self.n_items = n_items

        max_len = 0
        for seq in data[0]:
            if len(seq) > max_len:
                max_len = len(seq)
        self.max_len = max_len
        self.raw_sessions = np.asarray(data[0])
        self.raw_labs = np.asarray(data[1])
        self.length = len(self.raw_sessions)

    def __len__(self):
        return self.length

    def generate_batch(self, batch_size):
        n_batch = self.length // batch_size
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length - batch_size, self.length)
        return slices

    def get_slice_sess_mask(self, index):
        inp_sess = self.raw_sessions[index]
        targets = self.raw_labs[index]
        lengths = []
        for session in inp_sess:
            lengths.append(len(session))
        max_length = max(lengths)
        inp_sess, mask_1, mask_inf = self.zero_padding_mask(inp_sess, max_length)
        return inp_sess, targets, mask_1, mask_inf, lengths

    def zero_padding_mask(self, data, max_length):
        out_data = np.zeros((len(data), max_length), dtype=np.int)
        mask_1 = np.zeros((len(data), max_length), dtype=np.int)
        mask_inf = np.full((len(data), max_length), float('-inf'), dtype=np.float32)
        for i in range(len(data)):
            out_data[i, :len(data[i])] = data[i]
            mask_1[i, :len(data[i])] = 1
            mask_inf[i, :len(data[i])] = 0.0
        return out_data, mask_1, mask_inf


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    # np.random.shuffle(sidx)
    # print(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


########################################################################################################################
# 处理LightGCN的稀疏矩阵
def handle_dense_to_sp(np_A):
    norm_adj = getSparseGraph(np_A)
    tensor_adj = convert_spmat_to_sptensor(norm_adj)
    return tensor_adj


def getSparseGraph(np_A):
    print("generating adjacency matrix")
    s = time.time()
    csr_A = csr_matrix(np_A)
    adj_mat = sp.dok_matrix(csr_A, dtype=np.float32)

    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)

    norm_adj = d_mat.dot(adj_mat)
    norm_adj = norm_adj.dot(d_mat)
    norm_adj = norm_adj.tocsr()

    end = time.time()
    print(f"costing {end - s}s, saved norm_mat...")

    return norm_adj


def convert_spmat_to_sptensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))




