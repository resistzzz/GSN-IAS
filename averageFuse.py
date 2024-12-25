import os
import argparse
import pickle
import time
import datetime
from model_v1 import *
import numpy as np
from data import *
import warnings
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch_scatter import scatter
import random
warnings.filterwarnings("ignore")


def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='RetailRocket', help='yoochoose1_64/diginetica/sample')
'''训练基本参数'''
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_epoch', type=list, default=[3, 6, 9, 12], help='the epoch which the learning rate decay')
# parser.add_argument('--lr_dc_epoch', type=list, default=[5, 10, 15], help='the epoch which the learning rate decay')
parser.add_argument('--patience', type=int, default=5)

'''模型超参数'''
parser.add_argument('--anchor_num', type=int, default=50, help='number of cluster')
parser.add_argument('--anchor_method', default='infor', help='local/global')
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--routing_iter', type=int, default=4)
parser.add_argument('--hop', type=int, default=1)   # 1 or 2 or 3
parser.add_argument('--sample_num', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--topk', type=list, default=[20], help='topk recommendation')  # [5, 10, 20]

# parser.add_argument('--save_path', default='model_save', help='save model root path')
parser.add_argument('--save_path', default=None, help='save model root path')
parser.add_argument('--save_epochs', default=[3, 6, 9, 12], type=list)
# parser.add_argument('--save_epochs', default=[5, 10, 15], type=list)

opt = parser.parse_args()
print(opt)

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')


def main():
    t0 = time.time()
    init_seed(2021)

    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))

    adj_items = pickle.load(open('datasets/' + opt.dataset + '/adj_' + str(opt.n_sample) + '.pkl', 'rb'))
    weight_items = pickle.load(open('datasets/' + opt.dataset + '/num_' + str(opt.n_sample) + '.pkl', 'rb'))
    param = pickle.load(open('datasets/' + opt.dataset + '/parm' + '.pkl', 'rb'))

    num_items, max_length, item2idx, idx2items = param['num_items'], param['max_length'], param['item2idx'], param[
        'idx2item']

    anchors = pickle.load(open('anchors/' + opt.dataset + '/anchors_' +
                               opt.anchor_method + '_' + str(opt.anchor_num) + '.pkl', 'rb'))

    train_data = transfer2idx(train_data, item2idx)
    test_data = transfer2idx(test_data, item2idx)

    # 按每条session的最长长度进行padding
    train_data = Data(train_data, num_items)
    test_data = Data(test_data, num_items)

    train_slices = train_data.generate_batch(opt.batch_size)
    test_slices = test_data.generate_batch(opt.batch_size)

    adj_items, weight_items = handle_adj(adj_items, weight_items, num_items, opt.sample_num)

    if opt.dataset == 'yoochoose1_64':
        load_file = 'model_save/yoochoose1_64/2022-01-06-15-06-31/epoch-12.pt'   # K=100, hop=2
    elif opt.dataset == 'diginetica':
        load_file = 'model_save/diginetica/2022-01-08-06-41-29/epoch-6.pt'      # K=1000, hop=5
    elif opt.dataset == 'RetailRocket':
        load_file = 'model_save/RetailRocket/2022-01-06-22-51-59/epoch-6.pt'    # K=500, hop=2
    else:
        load_file = 'model_save/sample/2021-11-18-22-21-48/epoch-6.pt'

    model = torch.load(load_file)
    print(model)

    print('----------------')
    print('start predicting: ', datetime.datetime.now())
    hit_dic, mrr_dic = {}, {}
    hit_ite, mrr_ite = {}, {}
    hit_pop, mrr_pop = {}, {}
    for k in opt.topk:
        hit_dic[k] = []
        mrr_dic[k] = []
        hit_ite[k] = []
        mrr_ite[k] = []
        hit_pop[k] = []
        mrr_pop[k] = []

    tau = 1.0
    model.eval()
    for index in test_slices:
        tes_scores, tes_scores_item, tes_scores_pop, tes_targets = forward(model, index, test_data, tau)
        tes_scores = 0.5 * tes_scores_item + 0.5 * tes_scores_pop

        for k in opt.topk:
            predict = tes_scores.cpu().topk(k)[1]
            predict = predict.cpu()
            for pred, target in zip(predict, tes_targets.cpu()):
                hit_dic[k].append(np.isin(target - 1, pred))
                if len(np.where(pred == target - 1)[0]) == 0:
                    mrr_dic[k].append(0)
                else:
                    mrr_dic[k].append(1 / (np.where(pred == target - 1)[0][0] + 1))

        for k in opt.topk:
            predict = tes_scores_item.cpu().topk(k)[1]
            predict = predict.cpu()
            for pred, target in zip(predict, tes_targets.cpu()):
                hit_ite[k].append(np.isin(target - 1, pred))
                if len(np.where(pred == target - 1)[0]) == 0:
                    mrr_ite[k].append(0)
                else:
                    mrr_ite[k].append(1 / (np.where(pred == target - 1)[0][0] + 1))
        for k in opt.topk:
            predict = tes_scores_pop.cpu().topk(k)[1]
            predict = predict.cpu()
            for pred, target in zip(predict, tes_targets.cpu()):
                hit_pop[k].append(np.isin(target - 1, pred))
                if len(np.where(pred == target - 1)[0]) == 0:
                    mrr_pop[k].append(0)
                else:
                    mrr_pop[k].append(1 / (np.where(pred == target - 1)[0][0] + 1))

    for k in opt.topk:
        hit_dic[k] = np.mean(hit_dic[k]) * 100
        mrr_dic[k] = np.mean(mrr_dic[k]) * 100
        hit_ite[k] = np.mean(hit_ite[k]) * 100
        mrr_ite[k] = np.mean(mrr_ite[k]) * 100
        hit_pop[k] = np.mean(hit_pop[k]) * 100
        mrr_pop[k] = np.mean(mrr_pop[k]) * 100
        print('HitIte@%d:\t%0.4f %%\tMRRIte@%d:\t%0.4f %%\t' % (k, hit_ite[k], k, mrr_ite[k]))
        print('HitPop@%d:\t%0.4f %%\tMRRPop@%d:\t%0.4f %%\t' % (k, hit_pop[k], k, mrr_pop[k]))
        print('HitAvg@%d:\t%0.4f %%\tMRRAvg@%d:\t%0.4f %%\t' % (k, hit_dic[k], k, mrr_dic[k]))


def forward(model, index, data, tau):
    inp_sess, targets, mask_1, mask_inf, lengths = data.get_slice_sess_mask(index)

    inp_sess = torch.LongTensor(inp_sess).to(device)
    lengths = torch.LongTensor(lengths).to(device)
    targets = torch.LongTensor(targets).to(device)
    mask_1 = torch.FloatTensor(mask_1).to(device)
    mask_inf = torch.FloatTensor(mask_inf).to(device)

    # scores = model(inp_sess, lengths, mask_inf, mask_1, tau)
    # return scores, targets

    scores, scores_item, scores_pop = model(inp_sess, lengths, mask_inf, mask_1, tau)
    return scores, scores_item, scores_pop, targets




main()