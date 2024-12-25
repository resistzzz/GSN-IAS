
import os
import argparse
import pickle
import time
import datetime
# from model_pop import *
from model_v1 import *
import numpy as np
import sys
from data import *
import warnings
warnings.filterwarnings("ignore")


def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='yoochoose1_64/diginetica/sample')
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
parser.add_argument('--anchor_num', type=int, default=40, help='number of cluster')
parser.add_argument('--anchor_method', default='infor', help='local/global')
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--routing_iter', type=int, default=4)
parser.add_argument('--hop', type=int, default=1)   # 1 or 2 or 3
parser.add_argument('--sample_num', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--topk', type=list, default=[20], help='topk recommendation')  # [5, 10, 20]

parser.add_argument('--validation', default='test', help='validation/test')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--log', type=bool, default=True)

parser.add_argument('--save_path', default='model_save', help='save model root path')
# parser.add_argument('--save_path', default=None, help='save model root path')
parser.add_argument('--save_epochs', default=[3, 6, 9, 12], type=list)
# parser.add_argument('--save_epochs', default=[5, 10, 15], type=list)

opt = parser.parse_args()

if opt.log:
    path = 'log_v1/' + opt.dataset
    if not os.path.exists(path):
        os.makedirs(path)
    file = path + '/' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '_K' + str(opt.anchor_num) + \
           '_hop' + str(opt.hop) + '_' + opt.validation + '.txt'
    f = open(file, 'w')
else:
    f = sys.stdout

if opt.validation == 'test':
    print('Now is testset', file=f)
else:
    print('Now is validationset', file=f)


print(opt, file=f)


USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

if opt.save_path is not None and opt.dataset != 'sample':
    save_path = opt.save_path + '/' + opt.dataset
    save_dir = save_path + '/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    print('save dir: ', save_dir, file=f)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def main():
    t0 = time.time()
    init_seed(2021)

    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))

    if opt.validation == 'validation':
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))
    # test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))

    adj_matrix = pickle.load(open('datasets/' + opt.dataset + '/adj_matrix' + '.pkl', 'rb'))
    adj_matrix = adj_matrix[1:, 1:]
    sptensor_adj = handle_dense_to_sp(adj_matrix)

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

    model = LightGCNLC(opt, num_items, anchors, sptensor_adj, device=device)

    model = model.to(device)
    print(model, file=f)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_dc_epoch, gamma=opt.lr_dc)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=opt.lr_dc)

    best_result = {}
    best_epoch = {}
    for k in opt.topk:
        best_result[k] = [0, 0]
        best_epoch[k] = [0, 0]
    bad_counter = 0
    # tau = [1.0, 1.0, 1.0, 0.6, 0.6, 0.6, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    for epoch in range(opt.epochs):
        # tau = max(1.0 - epoch * 0.1, 0.1)
        tau = 1.0
        st = time.time()
        print('-------------------------------------------', file=f)
        print('tau: %0.2f' % tau, file=f)

        print('epoch: ', epoch, file=f)

        hit, mrr = train_test(model, train_data, test_data, train_slices, test_slices, optimizer, scheduler, tau)
        scheduler.step()

        if opt.save_path is not None and epoch in opt.save_epochs and opt.dataset != 'sample':
            save_file = save_dir + '/epoch-' + str(epoch) + '.pt'
            torch.save(model, save_file)
            print('save success! :)', file=f)
        bad_counter += 1

        for k in opt.topk:
            if hit[k] > best_result[k][0]:
                best_result[k][0] = hit[k]
                best_epoch[k][0] = epoch
                bad_counter = 0
            if mrr[k] > best_result[k][1]:
                best_result[k][1] = mrr[k]
                best_epoch[k][1] = epoch
                bad_counter = 0
            print('Hit@%d:\t%0.4f %%\tMRR@%d:\t%0.4f %%\t[%0.2f s]' % (k, hit[k], k, mrr[k], (time.time() - st)), file=f)
        if bad_counter > opt.patience:
            break

    print('------------------best result-------------------', file=f)
    for k in opt.topk:
        print('Best Result: Hit@%d: %0.4f %%\tMRR@%d: %0.4f %%\t[%0.2f s]' %
              (k, best_result[k][0], k, best_result[k][1], (time.time() - t0)), file=f)
        print('Best Epoch: Hit@%d: %d\tMRR@%d: %d\t[%0.2f s]' % (
            k, best_epoch[k][0], k, best_epoch[k][1], (time.time() - t0)), file=f)
    print('------------------------------------------------', file=f)
    print('Run time: %0.2f s' % (time.time() - t0), file=f)


def train_test(model, train_data, test_data, train_slices, test_slices, optimizer, scheduler, tau):
    print('start training: ', datetime.datetime.now(), file=f)
    model.train()
    # scheduler.step()
    total_loss = []
    total_loss_item = []
    total_loss_pop = []
    total_loss_con = []

    beta = 1.0
    for index in train_slices:
        optimizer.zero_grad()

        # scores, targets = forward(model, index, train_data, tau)
        # loss = model.loss_function(scores, targets - 1)

        scores, scores_item, scores_pop, targets = forward(model, index, train_data, tau)
        # scores, scores_item, scores_pop, con_loss, targets = forward(model, index, train_data, tau)

        loss = model.loss_function(scores, targets - 1)
        loss_item = model.loss_function1(scores_item, targets - 1)
        loss_pop = model.loss_function2(scores_pop, targets - 1)
        loss = loss + loss_item + loss_pop

        loss.backward()

        optimizer.step()

        total_loss.append(loss.item())
        total_loss_item.append(loss_item.item())
        total_loss_pop.append(loss_pop.item())
        # total_loss_con.append(con_loss.item())
    #
    print('Loss:\t%.8f\tlr:\t%0.8f' % (np.mean(total_loss), optimizer.state_dict()['param_groups'][0]['lr']), file=f)
    print('LossIte:\t%.8f\tlr:\t%0.8f' % (np.mean(total_loss_item), optimizer.state_dict()['param_groups'][0]['lr']), file=f)
    print('LossPop:\t%.8f\tlr:\t%0.8f' % (np.mean(total_loss_pop), optimizer.state_dict()['param_groups'][0]['lr']), file=f)
    # print('LossCon:\t%.8f\tlr:\t%0.8f' % (np.mean(total_loss_con), optimizer.state_dict()['param_groups'][0]['lr']))

    print('----------------', file=f)
    print('start predicting: ', datetime.datetime.now(), file=f)
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

    with torch.no_grad():
        model.eval()
        for index in test_slices:
            # tes_scores, tes_targets = forward(model, index, test_data, tau)
            tes_scores, tes_scores_item, tes_scores_pop, tes_targets = forward(model, index, test_data, tau)
            # tes_scores, tes_scores_item, tes_scores_pop, _, tes_targets = forward(model, index, test_data, tau)

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
        print('HitIte@%d:\t%0.4f %%\tMRRIte@%d:\t%0.4f %%\t' % (k, hit_ite[k], k, mrr_ite[k]), file=f)
        print('HitPop@%d:\t%0.4f %%\tMRRPop@%d:\t%0.4f %%\t' % (k, hit_pop[k], k, mrr_pop[k]), file=f)

    return hit_dic, mrr_dic


def forward(model, index, data, tau):
    inp_sess, targets, mask_1, mask_inf, lengths = data.get_slice_sess_mask(index)

    inp_sess = torch.LongTensor(inp_sess).to(device)
    lengths = torch.LongTensor(lengths).to(device)
    targets = torch.LongTensor(targets).to(device)
    mask_1 = torch.FloatTensor(mask_1).to(device)
    mask_inf = torch.FloatTensor(mask_inf).to(device)

    scores, scores_item, scores_pop = model(inp_sess, lengths, mask_inf, mask_1, tau)
    return scores, scores_item, scores_pop, targets

main()





